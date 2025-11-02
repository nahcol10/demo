import os
import sys  # Added missing import
import tempfile
import uuid
import shutil
import glob
import json
import logging

import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
import numpy as np

import model as ocr_model  # Imports from model.py

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)

# --- Load Model and Vocabulary (Global) ---
MODEL_PATH = 'ocr_model.h5'
VOCAB_PATH = 'vocabulary.json'
INFERENCE_MODEL = None
DECODE_WRAPPER = None
AUTOTUNE = tf.data.AUTOTUNE

def load_global_model():
    """Loads the inference model and vocabulary into memory."""
    global INFERENCE_MODEL, DECODE_WRAPPER, AUTOTUNE

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        log.error(f"Error: '{MODEL_PATH}' or '{VOCAB_PATH}' not found.")
        log.error("Please run train.py first to generate these files.")
        return False

    log.info("Loading vocabulary...")
    try:
        (
            char_to_num,
            num_to_char,
            n_classes,
            max_label_length,
        ) = ocr_model.load_vocabulary(VOCAB_PATH)
        log.info(f"Vocabulary loaded: {n_classes} unique characters, max length: {max_label_length}")
    except Exception as e:
        log.error(f"Failed to load vocabulary: {e}")
        return False

    log.info("Loading trained Keras model...")
    try:
        # Load the full training model (with 3 inputs)
        training_model = keras.models.load_model(
            MODEL_PATH, custom_objects={'CTCLayer': ocr_model.CTCLayer}
        )
        
        # Build the inference model (with only 1 input: the image)
        image_input = training_model.input[0]  # Get the 'image' input
        output_layer = training_model.get_layer(name='dense_1')  # Get the softmax output
        INFERENCE_MODEL = keras.Model(inputs=image_input, outputs=output_layer.output)
        
        # Create a decoder function wrapper
        def decode_wrapper(pred_label):
            return ocr_model.decode_pred(pred_label, num_to_char)
        
        DECODE_WRAPPER = decode_wrapper
        log.info("--- Model loaded successfully ---")
        INFERENCE_MODEL.summary()  # Show the inference model structure
        return True
    
    except Exception as e:
        log.error(f"Failed to load Keras model: {e}")
        return False

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict text from an uploaded PDF file.
    """
    if INFERENCE_MODEL is None or DECODE_WRAPPER is None:
        log.warning("Predict request received but model is not loaded")
        return jsonify({"error": "Model is not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    pdf_file = request.files["file"]
    if pdf_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if pdf_file and pdf_file.filename.lower().endswith(".pdf"):
        # Create a unique temporary directory for this request
        run_id = str(uuid.uuid4())
        temp_dir = os.path.join(tempfile.gettempdir(), run_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Define sub-directories
        pdf_path = os.path.join(temp_dir, pdf_file.filename)
        image_dir = os.path.join(temp_dir, "images")
        craft_dir = os.path.join(temp_dir, "craft_results")
        sorted_dir = os.path.join(temp_dir, "sorted_boxes")
        words_dir = os.path.join(temp_dir, "words")

        try:
            pdf_file.save(pdf_path)
            log.info(f"[{run_id}] PDF file saved to {pdf_path}")

            # --- 1. Preprocessing Pipeline ---
            log.info(f"[{run_id}] Starting preprocessing...")
            ocr_model.pdf_to_images(pdf_path, image_dir)
            if not ocr_model.run_craft_detection(image_dir, craft_dir):
                raise Exception("CRAFT detection failed")
            ocr_model.sort_bounding_boxes(craft_dir, sorted_dir)
            ocr_model.apply_extraction_to_folder_for_test(image_dir, sorted_dir, words_dir)
            ocr_model.pad_and_resize_images(words_dir)
            log.info(f"[{run_id}] Preprocessing complete.")

            # --- 2. Prepare Data for Inference ---
            word_files = glob.glob(os.path.join(words_dir, "*.png"))
            if not word_files:
                log.info(f"[{run_id}] No word images found, returning empty text")
                return jsonify({"text": ""})  # Return empty if no text found

            # Sort files based on the "word_index;line_index.png" naming
            sorted_words_info = []
            for wf in word_files:
                basename = os.path.basename(wf)
                parts = os.path.splitext(basename)[0].split(';')
                if len(parts) == 2:
                    try:
                        word_idx = int(parts[0])
                        line_idx = int(parts[1])
                        sorted_words_info.append((word_idx, line_idx, wf))
                    except ValueError:
                        continue
            
            # Sort by word index
            sorted_words_info.sort()
            sorted_paths = [path for _, _, path in sorted_words_info]

            log.info(f"[{run_id}] Found {len(sorted_paths)} word images for inference")

            # Create tf.data.Dataset (only images, no labels)
            def load_inference_image(path):
                return ocr_model.load_image(path)

            inference_ds = (
                tf.data.Dataset.from_tensor_slices(sorted_paths)
                .map(load_inference_image, num_parallel_calls=AUTOTUNE)
                .batch(32)  # Use a reasonable batch size for inference
                .prefetch(AUTOTUNE)
            )

            # --- 3. Run Inference ---
            log.info(f"[{run_id}] Running inference on {len(sorted_paths)} words...")
            predictions = INFERENCE_MODEL.predict(inference_ds, verbose=0)
            decoded_predictions = DECODE_WRAPPER(predictions)
            log.info(f"[{run_id}] Inference complete. Decoded {len(decoded_predictions)} predictions")

            # --- 4. Format Output (Reconstruct lines) ---
            output_lines = []
            current_line_idx = -1
            current_line_text = []

            for (word_idx, line_idx, _), text in zip(sorted_words_info, decoded_predictions):
                if line_idx != current_line_idx:
                    if current_line_text:
                        output_lines.append(" ".join(current_line_text))
                    current_line_text = [text]
                    current_line_idx = line_idx
                else:
                    current_line_text.append(text)
            
            # Add the last line
            if current_line_text:
                output_lines.append(" ".join(current_line_text))
            
            final_text = "\n".join(output_lines)
            log.info(f"[{run_id}] Final text length: {len(final_text)} characters")
            
            return jsonify({"text": final_text})

        except Exception as e:
            log.error(f"Error during prediction for {run_id}: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
        
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    log.info(f"[{run_id}] Cleaned up temp directory.")
                except Exception as e:
                    log.warning(f"[{run_id}] Failed to clean up temp directory: {e}")

    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    if INFERENCE_MODEL is not None and DECODE_WRAPPER is not None:
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model_path": MODEL_PATH,
            "vocab_path": VOCAB_PATH
        }), 200
    else:
        return jsonify({
            "status": "error", 
            "message": "Model not loaded",
            "model_loaded": False
        }), 503

@app.route("/", methods=["GET"])
def home():
    """Home endpoint with API information."""
    return jsonify({
        "api_name": "OCR PDF Text Extraction API",
        "endpoints": {
            "/predict": "POST - Upload PDF file to extract text",
            "/health": "GET - Check API and model status",
            "/": "GET - This information"
        },
        "model_status": "loaded" if INFERENCE_MODEL is not None else "not loaded"
    }), 200

if __name__ == "__main__":
    log.info("Starting OCR API server...")
    if not load_global_model():
        log.error("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    
    log.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)