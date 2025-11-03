import os
import sys
import tempfile
import uuid
import shutil
import json
import logging
import fitz  # PyMuPDF
import numpy as np
from flask import Flask, request, jsonify
import easyocr

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
app = Flask(__name__)

# --- Initialize EasyOCR Reader (Global) ---
EASYOCR_READER = None
SUPPORTED_LANGUAGES = ['ne', 'en']  # Nepali and English

def load_easyocr_reader():
    """Initializes the EasyOCR reader for Nepali and English languages."""
    global EASYOCR_READER
    try:
        log.info(f"Loading EasyOCR reader with languages: {SUPPORTED_LANGUAGES}")
        # Try GPU first, fall back to CPU if GPU not available
        try:
            EASYOCR_READER = easyocr.Reader(SUPPORTED_LANGUAGES, gpu=True)
            log.info("EasyOCR reader loaded successfully with GPU")
        except Exception as gpu_error:
            log.warning(f"GPU not available, falling back to CPU: {gpu_error}")
            EASYOCR_READER = easyocr.Reader(SUPPORTED_LANGUAGES, gpu=False)
            log.info("EasyOCR reader loaded successfully with CPU")
        return True
    except Exception as e:
        log.error(f"Failed to load EasyOCR reader: {e}")
        return False

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict text from an uploaded PDF file using EasyOCR.
    """
    global EASYOCR_READER
    
    if EASYOCR_READER is None:
        log.warning("Predict request received but EasyOCR reader is not loaded")
        return jsonify({"error": "OCR reader is not loaded."}), 500
        
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
        
        try:
            # Save the uploaded PDF
            pdf_file.save(pdf_path)
            log.info(f"[{run_id}] PDF file saved to {pdf_path}")
            
            # --- 1. Convert PDF to images ---
            log.info(f"[{run_id}] Converting PDF to images...")
            os.makedirs(image_dir, exist_ok=True)
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for i, page in enumerate(doc):
                # Get pixmap with higher DPI for better quality
                pix = page.get_pixmap(dpi=300)
                image_path = os.path.join(image_dir, f"page_{i+1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
                log.debug(f"[{run_id}] Saved page {i+1} to {image_path}")
            
            doc.close()
            log.info(f"[{run_id}] Converted {len(image_paths)} pages to images")
            
            # --- 2. Process each page with EasyOCR ---
            log.info(f"[{run_id}] Running EasyOCR on {len(image_paths)} pages...")
            all_results = []
            
            for page_num, image_path in enumerate(image_paths):
                log.info(f"[{run_id}] Processing page {page_num+1}/{len(image_paths)}")
                try:
                    # EasyOCR can directly read from file path
                    results = EASYOCR_READER.readtext(
                        image_path, 
                        detail=1, 
                        paragraph=False,
                        batch_size=4  # Process in batches for better performance
                    )
                    all_results.append((page_num+1, results))
                    log.info(f"[{run_id}] Page {page_num+1}: Found {len(results)} text elements")
                except Exception as page_error:
                    log.error(f"[{run_id}] Error processing page {page_num+1}: {page_error}")
                    # Continue with other pages even if one fails
                    all_results.append((page_num+1, []))
            
            # --- 3. Format the output text ---
            final_text = ""
            for page_num, page_results in all_results:
                if not page_results:
                    log.warning(f"[{run_id}] No text found on page {page_num}")
                    if page_num > 1:
                        final_text += f"\n--- Page {page_num} (No text detected) ---\n"
                    else:
                        final_text += "No text detected on page 1\n"
                    continue
                
                # Sort results by y-coordinate of bounding box (top to bottom)
                sorted_results = sorted(page_results, key=lambda x: x[0][0][1])
                
                # Group text by lines based on y-coordinate proximity
                lines = []
                current_line = []
                last_y = None
                line_threshold = 20  # Adjust this value based on your PDFs
                
                for result in sorted_results:
                    bbox = result[0]
                    text = result[1]
                    confidence = result[2]
                    
                    # Get average y-coordinate of the bounding box
                    avg_y = sum([point[1] for point in bbox]) / 4
                    
                    # Determine if this belongs to a new line
                    if last_y is None or abs(avg_y - last_y) > line_threshold:
                        if current_line:
                            # Sort current line by x-coordinate before adding
                            current_line_sorted = sorted(current_line, key=lambda x: x[0][0][0])
                            lines.append(current_line_sorted)
                        current_line = [result]
                        last_y = avg_y
                    else:
                        current_line.append(result)
                
                # Don't forget the last line
                if current_line:
                    current_line_sorted = sorted(current_line, key=lambda x: x[0][0][0])
                    lines.append(current_line_sorted)
                
                # Build page text from lines
                page_text = ""
                for line in lines:
                    line_text = " ".join([item[1] for item in line])
                    page_text += line_text + "\n"
                
                if page_num > 1:
                    final_text += f"\n--- Page {page_num} ---\n"
                final_text += page_text.strip() + "\n"
            
            final_text = final_text.strip()
            if not final_text:
                final_text = "No text could be extracted from the PDF. This might be due to:\n- Scanned PDF (image-based)\n- Poor image quality\n- Unsupported language\n- Complex layout"
            
            log.info(f"[{run_id}] Extraction completed. Text length: {len(final_text)} characters")
            
            return jsonify({
                "text": final_text,
                "pages_processed": len(image_paths),
                "status": "success"
            })
            
        except Exception as e:
            log.error(f"Error during prediction for {run_id}: {str(e)}", exc_info=True)
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500
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
    if EASYOCR_READER is not None:
        return jsonify({
            "status": "ok",
            "ocr_reader_loaded": True,
            "languages": SUPPORTED_LANGUAGES
        }), 200
    else:
        return jsonify({
            "status": "error", 
            "message": "OCR reader not loaded",
            "ocr_reader_loaded": False
        }), 503

@app.route("/", methods=["GET"])
def home():
    """Home endpoint with API information."""
    return jsonify({
        "api_name": "OCR PDF Text Extraction API (EasyOCR with Nepali Support)",
        "endpoints": {
            "/predict": "POST - Upload PDF file to extract text",
            "/health": "GET - Check API and OCR reader status",
            "/": "GET - This information"
        },
        "ocr_status": "loaded" if EASYOCR_READER is not None else "not loaded",
        "supported_languages": SUPPORTED_LANGUAGES
    }), 200

if __name__ == "__main__":
    log.info("Starting OCR API server with EasyOCR...")
    if not load_easyocr_reader():
        log.error("Failed to load EasyOCR reader. Exiting.")
        sys.exit(1)
        
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    log.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)