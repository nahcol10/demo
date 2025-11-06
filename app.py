import os
import sys
import uuid
import shutil
import tempfile
import logging
import json
import fitz  # PyMuPDF
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, send_from_directory
import cv2
import easyocr
# PIL is no longer needed for preprocessing
# from PIL import Image, ImageOps

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# === TENSORFLOW MEMORY FIX ===
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Enabled memory growth for {len(gpus)} GPUs.")
    except RuntimeError as e:
        logging.error(f"Error setting memory growth: {e}")
# === END FIX ===

log = logging.getLogger(__name__)
app = Flask(__name__)

# --- Constants ---
# These must match model.py
IMG_HEIGHT = 50
IMG_WIDTH = 200
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH) # (50, 200)
CHANNELS = 1

# --- Model Globals ---
MODEL_PATH = os.environ.get('OCR_MODEL_PATH', 'ocr_model.h5')
VOCAB_PATH = os.environ.get('OCR_VOCAB_PATH', 'vocabulary.json')
INFERENCE_MODEL = None
NUM_TO_CHAR = None
EASYOCR_READER = None
MODEL_METADATA = {}


# ===================================================================
# === FIX: PREPROCESSING FUNCTION ===
# This function now EXACTLY matches model.py's load_image function
# ===================================================================
def load_and_preprocess_image(image_np, image_size=IMAGE_SIZE, channels=CHANNELS):
    """
    Loads and preprocesses a single image from memory,
    matching the training pipeline exactly.
    """
    try:
        # Convert numpy array (from cv2) to a TensorFlow tensor
        # image_np is already grayscale from the predict function
        image = tf.convert_to_tensor(image_np, dtype=tf.float32)
        
        # Add channel dimension if it's missing
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)

        # Normalize (assuming model.py's load_image did this via convert_image_dtype)
        # If it was already 0-255, we normalize.
        if image.dtype == tf.float32 and tf.reduce_max(image) > 1.0:
             image = image / 255.0
        elif image.dtype != tf.float32:
             image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)

        # Resize to (IMG_HEIGHT, IMG_WIDTH)
        resized_image = tf.image.resize(images=image, size=image_size)
        
        # Transpose to (IMG_WIDTH, IMG_HEIGHT, 1)
        target_img = tf.transpose(resized_image, perm=[1, 0, 2])
        target_img = tf.cast(target_img, dtype=tf.float32)
        
        return target_img
        
    except Exception as e:
        log.warning(f"Skipping image due to preprocessing error: {e}")
        return None
# ===================================================================
# === END FIX ===
# ===================================================================

def decode_batch_predictions(pred, num_to_char, max_label_length):
    """Decodes a batch of predictions from the CTC layer."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_label_length
    ]
    
    # Decode numbers to characters
    output_texts = []
    for res in results:
        res = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_texts.append(res)
    return output_texts

def load_inference_model(model_path, vocab_path):
    """Loads the trained model and vocabulary."""
    global MODEL_METADATA, NUM_TO_CHAR
    
    log.info(f"Loading vocabulary from {vocab_path}...")
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            MODEL_METADATA = json.load(f)
        
        vocabulary = MODEL_METADATA['vocabulary']
        
        # Create the character-to-number mapping
        NUM_TO_CHAR = tf.keras.layers.StringLookup(
            vocabulary=vocabulary, mask_token=None, invert=True
        )
        
    except FileNotFoundError:
        log.error(f"Vocabulary file not found: {vocab_path}")
        return None, None
    except json.JSONDecodeError:
        log.error(f"Error decoding JSON from {vocab_path}")
        return None, None
    
    log.info(f"Loading model from {model_path}...")
    try:
        inference_model = keras.models.load_model(model_path)
        
        log.info("Model loaded successfully.")
        return inference_model, NUM_TO_CHAR
        
    except FileNotFoundError:
        log.error(f"Model file not found: {model_path}")
        return None, None
    except Exception as e:
        log.error(f"An error occurred while loading the model: {e}", exc_info=True)
        return None, None

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload, PDF conversion, text detection, and recognition."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_pdf_path = os.path.join(temp_dir, str(uuid.uuid4()) + ".pdf")
                file.save(temp_pdf_path)
                
                log.info(f"File saved to {temp_pdf_path}")
                
                # --- 1. PDF to Images ---
                doc = fitz.open(temp_pdf_path)
                images = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=300)
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4: # RGBA
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                    elif pix.n == 1: # Grayscale
                        img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                    images.append(img_data)
                doc.close()
                log.info(f"Converted PDF to {len(images)} images.")
                
                # --- 2. Extract Words (using EasyOCR) ---
                log.info("Extracting words using EasyOCR...")
                word_images = []
                
                if EASYOCR_READER is None:
                    log.error("EasyOCR reader is not initialized.")
                    return jsonify({'error': 'EasyOCR reader not loaded on server'}), 500
                
                for page_image in images:
                    # page_image is a numpy array (BGR)
                    # We run readtext with paragraph=False to get individual words
                    results = EASYOCR_READER.readtext(page_image, detail=1, paragraph=False)
                    for (bbox, text, prob) in results:
                        (tl, tr, br, bl) = bbox
                        tl = (int(tl[0]), int(tl[1]))
                        br = (int(br[0]), int(br[1]))
                        cropped_word_bgr = page_image[tl[1]:br[1], tl[0]:br[0]]
                        if cropped_word_bgr.size > 0:
                            cropped_word_gray = cv2.cvtColor(cropped_word_bgr, cv2.COLOR_BGR2GRAY)
                            word_images.append(cropped_word_gray)
                
                log.info(f"Extracted {len(word_images)} words.")
                if not word_images:
                     return jsonify({'error': 'No text detected in the document.'}), 400

                # --- 3. Preprocess for Model ---
                processed_images = []
                for img in word_images:
                    # Pass the grayscale numpy array to the new function
                    processed = load_and_preprocess_image(img)
                    if processed is not None:
                        processed_images.append(processed)
                
                if not processed_images:
                    return jsonify({'error': 'Text was detected but could not be processed.'}), 400

                batch = tf.stack(processed_images, axis=0)
                log.info(f"Processed batch for model with shape: {batch.shape}")

                # --- 4. Predict ---
                if INFERENCE_MODEL is None:
                    log.error("Model is not loaded.")
                    return jsonify({'error': 'Model not loaded on server'}), 500
                
                predictions = INFERENCE_MODEL.predict(batch)
                
                # --- 5. Decode ---
                decoded_texts = decode_batch_predictions(
                    predictions, 
                    NUM_TO_CHAR, 
                    MODEL_METADATA.get('max_label_length', 50)
                )
                
                log.info(f"Decoded {len(decoded_texts)} predictions.")
                
                full_text = " ".join(decoded_texts)
                
                # --- Return correct UTF-8 JSON ---
                response_data = {
                    'id': str(uuid.uuid4()),
                    'filename': file.filename,
                    'word_count': len(decoded_texts),
                    'full_text': full_text,
                    'words': decoded_texts
                }
                response_json = json.dumps(response_data, ensure_ascii=False)
                response = app.response_class(
                    response_json,
                    mimetype='application/json; charset=utf-8'
                )
                return response

        except Exception as e:
            log.error(f"Error during prediction: {e}", exc_info=True)
            return jsonify({'error': f'An internal error occurred: {e}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok', 
        'model_loaded': INFERENCE_MODEL is not None,
        'vocabulary_loaded': NUM_TO_CHAR is not None,
        'easyocr_loaded': EASYOCR_READER is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    # Load TF model on startup
    INFERENCE_MODEL, NUM_TO_CHAR = load_inference_model(MODEL_PATH, VOCAB_PATH)
    
    log.info("Loading EasyOCR reader...")
    try:
        # Use the 'db' detector, which is often better for non-linear text
        EASYOCR_READER = easyocr.Reader(['ne', 'en'], detector='db')
        log.info("EasyOCR reader loaded successfully (using 'db' detector).")
    except Exception as e:
        log.error(f"Failed to load EasyOCR reader: {e}")
    
    if INFERENCE_MODEL is None or NUM_TO_CHAR is None:
        log.warning("=" * 50)
        log.warning("WARNING: Model or vocabulary failed to load.")
        log.warning("The /predict endpoint may not work.")
        log.warning("=" * 50)
    
    log.info(f"Starting Flask server on port {port}...")
    app.run(debug=False, host='0.0.0.0', port=port)