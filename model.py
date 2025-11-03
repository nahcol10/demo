#!/usr/bin/env python
# coding: utf-8

# --- IMPORTS ---
import os
import subprocess
import sys
import shutil
import zipfile
import glob
import csv
import re
import gdown
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
from tqdm import tqdm
import json
import logging
import math
from typing import List, Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- CONSTANTS ---
IMG_WIDTH = 200
IMG_HEIGHT = 50
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRAFT_DIR = os.path.join(BASE_DIR, 'CRAFT-pytorch')
CRAFT_TEST_SCRIPT = os.path.join(CRAFT_DIR, 'test.py')
CRAFT_WEIGHTS_DIR = os.path.join(CRAFT_DIR, 'weights')
CRAFT_WEIGHTS_FILE = os.path.join(CRAFT_WEIGHTS_DIR, 'craft_mlt_25k.pth')

# --- Keras Model Definition ---

class CTCLayer(layers.Layer):
    def __init__(self, name="ctc_layer", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        y_true, y_pred, label_len_tensor = inputs

        # y_true: (batch_size, max_label_len) → must be int32
        y_true = tf.cast(y_true, tf.int32)

        # y_pred: (batch_size, time_steps, num_classes)
        batch_size = tf.shape(y_pred)[0]
        input_len = tf.shape(y_pred)[1]  # e.g., 50

        # input_length: (batch_size, 1) — all sequences are full length
        input_len_tensor = tf.fill((batch_size, 1), input_len)

        # label_len_tensor is already (batch_size, 1), just ensure int32
        label_len_tensor = tf.cast(label_len_tensor, tf.int32)

        # Compute CTC loss
        loss = keras.backend.ctc_batch_cost(
            y_true=y_true,
            y_pred=y_pred,
            input_length=input_len_tensor,
            label_length=label_len_tensor
        )

        self.add_loss(loss)
        return y_pred  # must return y_pred for model output

    def get_config(self):
        return super().get_config()

def build_ocr_model(n_classes: int, char_to_num_vocab_len: int) -> keras.Model:
    """
    Builds the complete CNN + RNN + CTC model for training.
    
    Args:
        n_classes: Number of character classes
        char_to_num_vocab_len: Length of vocabulary including [UNK] token
    
    Returns:
        Compiled Keras model
    """
    input_images = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image")
    
    # Set correct dtypes for labels
    target_labels = layers.Input(shape=(None,), name="label", dtype="int64")
    target_labels_len = layers.Input(shape=(1,), name="label_length", dtype="int32") 

    # CNN Network
    x = layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='relu', kernel_initializer='he_normal')(input_images)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), (1, 1), 'same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Encoding Space
    new_shape = ((IMG_WIDTH // 4), (IMG_HEIGHT // 4) * 128)
    encoding = layers.Reshape(target_shape=new_shape)(x)
    encoding = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(encoding)
    encoding = layers.Dropout(0.5)(encoding)

    # RNN Network
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(encoding)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.5))(x)

    # Output Layer
    output = layers.Dense(char_to_num_vocab_len + 1, activation='softmax', name='dense_1')(x)

    # CTC Layer
    ctc_layer_output = CTCLayer()([target_labels, output, target_labels_len]) 

    # Model
    model = keras.Model(
        inputs=[input_images, target_labels, target_labels_len], 
        outputs=[ctc_layer_output], 
        name="functional_ocr"
    )
    return model

# --- Data Loading and Decoding Functions ---

def load_image(image_path: str) -> tf.Tensor:
    """
    Loads, decodes, resizes, and transposes an image for the model.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image tensor
    """
    try:
        image = tf.io.read_file(image_path)
        decoded_image = tf.image.decode_png(contents=image, channels=1)  # Use decode_png for PNG files
        cnvt_image = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
        resized_image = tf.image.resize(images=cnvt_image, size=(IMG_HEIGHT, IMG_WIDTH))
        image = tf.transpose(resized_image, perm=[1, 0, 2])
        image = tf.cast(image, dtype=tf.float32)
        return image
    except Exception as e:
        log.error(f"Error loading image {image_path}: {e}")
        # Return a blank image as fallback
        return tf.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32)

def encode_single_sample(image_path: str, label: str, max_label_length: int, 
                        char_to_num_lookup: layers.StringLookup, num_classes: int) -> Dict[str, tf.Tensor]:
    """
    Encodes a single image/label pair into tensors for the model.
    
    Args:
        image_path: Path to image file
        label: Text label
        max_label_length: Maximum label length for padding
        char_to_num_lookup: StringLookup layer for character encoding
        num_classes: Number of character classes
    
    Returns:
        Dictionary with encoded tensors
    """
    image = load_image(image_path)
    chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
    vecs = char_to_num_lookup(chars)
    
    orig_len = tf.shape(vecs)[0]
    
    # --- START FIX ---
    
    # Define the model's time steps (from CNN architecture)
    time_steps = IMG_WIDTH // 4  # This is 50
    
    # 1. Cap the *reported length* at the number of time steps.
    #    CTC loss fails if label_len > input_len.
    capped_len = tf.minimum(orig_len, time_steps)
    
    # 2. Handle empty labels (after capping)
    if tf.equal(capped_len, 0):
        final_len = tf.constant(1, dtype=tf.int32)
        # Use [UNK] token for empty label
        vecs = tf.constant([0], dtype=tf.int64)
    else:
        final_len = tf.cast(capped_len, tf.int32)
        # Truncate 'vecs' to the capped length *before* padding
        vecs = tf.slice(vecs, [0], [final_len])

    # 3. Pad 'vecs' to max_label_length (for batching)
    pad_size = max_label_length - final_len
    
    if tf.greater(pad_size, 0):
        # Pad with 0 ([UNK])
        vecs = tf.pad(vecs, paddings=[[0, pad_size]], constant_values=0)
    elif tf.less(pad_size, 0):
        # This can happen if max_label_length < final_len
        # Truncate to max_label_length
        vecs = tf.slice(vecs, [0], [max_label_length])
            
    # Expand scalar shape () to (1,) to match Input layer
    final_len_expanded = tf.expand_dims(final_len, axis=-1)
        
    # Return a dictionary matching the model's input names
    return {
        'image': image, 
        'label': vecs, 
        'label_length': final_len_expanded # This is now capped at 50 (or 1)
    }

def decode_pred(pred_label: np.ndarray, num_to_char_lookup: layers.StringLookup) -> List[str]:
    """
    Decodes the predicted labels from the OCR model.
    
    Args:
        pred_label: Model predictions tensor
        num_to_char_lookup: Inverted StringLookup layer
    
    Returns:
        List of decoded text strings
    """
    try:
        input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
        # ctc_decode returns sparse tensors, which are int64 by default
        decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0]
        chars = num_to_char_lookup(decode)
        texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
        # Filter out [UNK] tokens, which are at index 0
        filtered_texts = [text.replace('[UNK]', "").strip() for text in texts]
        return filtered_texts
    except Exception as e:
        log.error(f"Error decoding predictions: {e}")
        return [""] * pred_label.shape[0]

# --- Vocabulary Management ---

def save_vocabulary(path: str, unique_chars: List[str], max_label_length: int) -> None:
    """
    Saves the character list and max length to a JSON file.
    
    Args:
        path: Output file path
        unique_chars: List of unique characters
        max_label_length: Maximum label length
    """
    # Keras StringLookup reserves 0 for OOV ([UNK])
    # The vocabulary list itself should not contain [UNK]
    vocab_data = {
        'vocabulary': list(unique_chars), 
        'max_label_length': int(max_label_length)
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    log.info(f"Vocabulary saved to {path} - {len(unique_chars)} characters, max length: {max_label_length}")

def load_vocabulary(path: str) -> Optional[Tuple[layers.StringLookup, layers.StringLookup, int, int]]:
    """
    Loads vocabulary and model parameters from JSON file.
    
    Args:
        path: Path to vocabulary JSON file
    
    Returns:
        Tuple of (char_to_num, num_to_char, n_classes, max_label_length) or None if failed
    """
    if not os.path.exists(path):
        log.error(f"Vocabulary file not found: {path}")
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        unique_chars = vocab_data['vocabulary']
        max_label_length = vocab_data['max_label_length']
        n_classes = len(unique_chars)
        
        # oov_token='[UNK]' ensures that index 0 is reserved for [UNK]
        char_to_num = layers.StringLookup(vocabulary=unique_chars, mask_token=None, oov_token='[UNK]', name='char_to_num')
        # The inverted lookup will map 0 back to [UNK]
        num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True, name='num_to_char')
        
        log.info(f"Vocabulary loaded: {n_classes} unique characters, max length: {max_label_length}")
        return char_to_num, num_to_char, n_classes, max_label_length
    except Exception as e:
        log.error(f"Error loading vocabulary from {path}: {e}")
        return None

# --- Preprocessing & Utility Functions ---

def _clone_craft_repo():
    """Clones the official CRAFT repository."""
    log.info("Cloning official CRAFT repository...")
    try:
        # Use the official CRAFT repository from ClovaAI
        if os.path.exists(CRAFT_DIR):
            shutil.rmtree(CRAFT_DIR)
            
        # Clone the repository
        subprocess.run(['git', 'clone', 'https://github.com/clovaai/CRAFT-pytorch.git', CRAFT_DIR], 
                      check=True, timeout=300)
        
        # Install required dependencies
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', os.path.join(CRAFT_DIR, 'requirements.txt')],
                      check=True, timeout=300)
        
        log.info("Official CRAFT repository cloned and dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to clone CRAFT repo: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected error cloning CRAFT repo: {e}")
        raise

def _download_craft_weights():
    """Downloads CRAFT weights."""
    os.makedirs(CRAFT_WEIGHTS_DIR, exist_ok=True)
    log.info("Downloading CRAFT weights (craft_mlt_25k.pth)...")
    try:
        # Use the correct Google Drive file ID for the General CRAFT model
        # From official repo: https://github.com/clovaai/CRAFT-pytorch
        file_id = '1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ'
        
        # Try gdown with the correct file ID
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, CRAFT_WEIGHTS_FILE, quiet=False, fuzzy=True)
        except Exception as gdown_error:
            log.warning(f"gdown method failed: {gdown_error}")
            # Fallback to direct download method
            try:
                log.info("Trying gdown.download with direct ID...")
                gdown.download(id=file_id, output=CRAFT_WEIGHTS_FILE, quiet=False)
            except Exception as e2:
                log.error(f"All download methods failed: {e2}")
                raise Exception(
                    f"Failed to download CRAFT weights. Please manually download "
                    f"the General model from https://drive.google.com/file/d/{file_id}/view "
                    f"and place it at {CRAFT_WEIGHTS_FILE}"
                )
        
        if not os.path.exists(CRAFT_WEIGHTS_FILE) or os.path.getsize(CRAFT_WEIGHTS_FILE) < 1000000:  # Should be > 1MB
            raise FileNotFoundError(
                f"CRAFT weights download incomplete or failed. Please manually download "
                f"from https://drive.google.com/file/d/{file_id}/view "
                f"and place it at {CRAFT_WEIGHTS_FILE}"
            )
        log.info(f"CRAFT weights downloaded successfully ({os.path.getsize(CRAFT_WEIGHTS_FILE)} bytes)")
    except Exception as e:
        log.error(f"Error downloading CRAFT weights: {e}")
        raise

def setup_craft_model():
    """Clones the CRAFT repo and downloads weights if they don't exist."""
    try:
        if not os.path.exists(CRAFT_DIR):
            _clone_craft_repo()
        else:
            log.info("CRAFT_Model directory already exists.")
        
        if not os.path.exists(CRAFT_WEIGHTS_FILE):
            _download_craft_weights()
        else:
            log.info("CRAFT weights already exist.")
            
        # Verify the test script exists
        if not os.path.exists(CRAFT_TEST_SCRIPT):
            log.error(f"CRAFT test script not found at: {CRAFT_TEST_SCRIPT}")
            raise FileNotFoundError("CRAFT test script missing")
            
    except Exception as e:
        log.error(f"Failed to setup CRAFT model: {e}")
        raise

def run_craft_detection(image_folder: str, result_folder: str) -> bool:
    """Runs the CRAFT test.py script on a folder of images."""
    try:
        log.info("Setting up CRAFT model...")
        setup_craft_model()  # Ensure CRAFT is ready
        
        os.makedirs(result_folder, exist_ok=True)
        
        # Determine CUDA availability
        has_cuda = tf.config.list_physical_devices('GPU')
        cuda_flag = 'True' if has_cuda else 'False'
        log.info(f"CUDA available: {has_cuda}")
        
        # CRAFT test.py uses a hardcoded result folder './result/', so we need to run it
        # from a specific directory and then move the results
        # Updated command for official CRAFT repository (without unsupported arguments)
        craft_command = [
            sys.executable,
            CRAFT_TEST_SCRIPT,
            '--trained_model', os.path.abspath(CRAFT_WEIGHTS_FILE),
            '--test_folder', os.path.abspath(image_folder),
            '--cuda', cuda_flag,
            '--poly'
        ]
        
        log.info(f"Running CRAFT text detection on {image_folder}...")
        log.info(f"Command: {' '.join(craft_command)}")
        
        # Run CRAFT from its directory
        craft_cwd = CRAFT_DIR
        result = subprocess.run(
            craft_command, 
            cwd=craft_cwd, 
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        log.info("CRAFT stdout:")
        log.info(result.stdout)
        if result.stderr:
            log.warning("CRAFT stderr:")
            log.warning(result.stderr)
            
        # Move results from CRAFT's default ./result/ to our result_folder
        craft_result_dir = os.path.join(CRAFT_DIR, 'result')
        if os.path.exists(craft_result_dir):
            # Move all result files to our desired location
            for file in glob.glob(os.path.join(craft_result_dir, '*')):
                dest_file = os.path.join(result_folder, os.path.basename(file))
                shutil.move(file, dest_file)
                log.debug(f"Moved {file} to {dest_file}")
        
        log.info(f"CRAFT processing complete. Results in {result_folder}")
        
        # Verify that result files were created
        result_files = glob.glob(os.path.join(result_folder, "*.txt"))
        if not result_files:
            log.error("No result files were generated by CRAFT")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        log.error(f"CRAFT detection failed with return code {e.returncode}")
        log.error(f"CRAFT stdout: {e.stdout}")
        log.error(f"CRAFT stderr: {e.stderr}")
        return False
    except subprocess.TimeoutExpired as e:
        log.error(f"CRAFT detection timed out after {e.timeout} seconds")
        return False
    except FileNotFoundError as e:
        log.error(f"CRAFT files not found: {e}")
        log.error(f"CRAFT_DIR: {CRAFT_DIR}")
        log.error(f"CRAFT_TEST_SCRIPT: {CRAFT_TEST_SCRIPT}")
        log.error(f"CRAFT_WEIGHTS_FILE: {CRAFT_WEIGHTS_FILE}")
        return False
    except Exception as e:
        log.error(f"Unexpected error running CRAFT: {e}", exc_info=True)
        return False

def pdf_to_images(pdf_path: str, output_folder: str) -> None:
    """Converts each page of a PDF file into a PNG image."""
    log.info(f"Converting {pdf_path} to images...")
    os.makedirs(output_folder, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)  # Store this BEFORE closing the document
        log.info(f"PDF opened successfully. Total pages: {total_pages}")
        for i, page in enumerate(tqdm(doc, desc="Converting PDF pages")):
            # Get pixmap with higher DPI for better quality
            pix = page.get_pixmap(dpi=300)
            output_path = os.path.join(output_folder, f"page_{i+1}.png")
            pix.save(output_path)
            log.debug(f"Saved page {i+1} to {output_path}")
        doc.close()
        log.info(f"Converted {total_pages} pages to images in {output_folder}")
    except Exception as e:
        log.error(f"Error converting PDF {pdf_path}: {e}")
        # Ensure document is closed even if an exception occurs
        if 'doc' in locals() and hasattr(doc, 'is_closed') and not doc.is_closed:
            doc.close()
        raise

def _parse_box_line(line: str) -> Optional[Tuple[float, int, str]]:
    """
    Parses a line from CRAFT output file and extracts coordinates.
    
    Args:
        line: Line from CRAFT output file
    
    Returns:
        Tuple of (avg_y, x_coord, original_line) or None if invalid
    """
    coords_str = line.strip().split(',')
    if len(coords_str) != 8:
        return None
    
    try:
        coords = [float(c) for c in coords_str]
        avg_y = (coords[1] + coords[3] + coords[5] + coords[7]) / 4  # Average of all y coordinates
        x_coord = min(coords[0], coords[2], coords[4], coords[6])  # Leftmost x coordinate
        return (avg_y, int(x_coord), line.strip())
    except (ValueError, IndexError) as e:
        log.warning(f"Error parsing box line: {e}")
        return None

def sort_bounding_boxes(bound_box_applied_folder: str, bound_box_sorted_folder: str) -> None:
    """Reads CRAFT's output text files, sorts the boxes, and saves to new files."""
    log.info("Sorting bounding boxes...")
    os.makedirs(bound_box_sorted_folder, exist_ok=True)
    
    craft_files = glob.glob(os.path.join(bound_box_applied_folder, "res_*.txt"))
    log.info(f"Found {len(craft_files)} CRAFT result files")
    
    for file_path in tqdm(craft_files, desc="Sorting Boxes"):
        boxes = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parsed = _parse_box_line(line)
                    if parsed:
                        boxes.append(parsed)
            
            # Sort by Y (primary) then X (secondary)
            boxes.sort(key=lambda x: (x[0], x[1]))
            
            # Write sorted boxes to new file
            base_name = os.path.basename(file_path)
            sorted_name = base_name.replace(".txt", "_sorted.txt")
            output_path = os.path.join(bound_box_sorted_folder, sorted_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for _, _, original_line in boxes:
                    f.write(original_line + '\n')
                    
            log.debug(f"Sorted {len(boxes)} boxes from {base_name}")
                    
        except Exception as e:
            log.warning(f"Error processing {file_path}: {e}")
            
    log.info(f"Sorted box files saved to {bound_box_sorted_folder}")

def _get_line_number(current_y: float, last_y: float, last_h: float, line_spacing_threshold: float = 0.7) -> int:
    """
    Determines if current box belongs to a new line based on vertical position.
    
    Args:
        current_y: Current box y-coordinate
        last_y: Previous box y-coordinate
        last_h: Previous box height
        line_spacing_threshold: Threshold for line spacing detection
    
    Returns:
        1 if new line detected, 0 otherwise
    """
    if last_y == -1:
        return 0
    
    vertical_gap = current_y - (last_y + last_h)
    return 1 if vertical_gap > (last_h * line_spacing_threshold) else 0

def extract_bounding_boxes(image_path: str, bounding_box_path: str, output_folder: str, 
                          word_counter_start: int = 0) -> int:
    """
    Extracts word images based on sorted bounding boxes.
    
    Args:
        image_path: Path to source image
        bounding_box_path: Path to bounding box file
        output_folder: Output folder for word images
        word_counter_start: Starting index for word counter
    
    Returns:
        Next word counter value
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            log.warning(f"Failed to read image {image_path}")
            return word_counter_start
            
        img_height, img_width = img.shape[:2]
        line_counter = 0
        word_counter = word_counter_start
        last_y = -1
        last_h = 0

        with open(bounding_box_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        log.debug(f"Processing {len(lines)} bounding boxes from {bounding_box_path}")
        
        for line_idx, line in enumerate(lines):
            try:
                coords = [int(float(c)) for c in line.strip().split(',')]
                if len(coords) != 8:
                    continue
                
                # Get bounding rectangle from coordinates
                points = np.array(coords).reshape(4, 2).astype(np.float32)
                rect = cv2.minAreaRect(points)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                x, y, w, h = cv2.boundingRect(box)

                # Validate coordinates
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                    continue

                # Add margin for better word recognition
                margin = max(2, min(5, int(min(w, h) * 0.1)))  # Adaptive margin
                y_start = max(0, y - margin)
                y_end = min(img_height, y + h + margin)
                x_start = max(0, x - margin)
                x_end = min(img_width, x + w + margin)
                
                cropped = img[y_start:y_end, x_start:x_end]
                
                if cropped.size == 0:
                    continue

                # Check if this is a new line
                if _get_line_number(y, last_y, last_h):
                    line_counter += 1
                
                filename = f"{word_counter};{line_counter}.png"
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped)
                
                word_counter += 1
                last_y = y
                last_h = h
                
            except Exception as e:
                log.warning(f"Error processing box {line_idx} in {bounding_box_path}: {e}")
                continue
                
        log.info(f"Extracted {word_counter - word_counter_start} words from {image_path}")
        return word_counter
        
    except Exception as e:
        log.error(f"Error in extract_bounding_boxes: {e}")
        return word_counter_start

def apply_extraction_to_folder_for_test(image_folder: str, bounding_box_folder: str, 
                                      output_folder: str) -> int:
    """
    Processes images, finds their sorted bounding box files, and extracts words.
    
    Args:
        image_folder: Folder containing source images
        bounding_box_folder: Folder containing sorted bounding box files
        output_folder: Output folder for extracted words
    
    Returns:
        Total number of words extracted
    """
    log.info(f"Extracting words to {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    log.info(f"Found {len(image_files)} images to process")

    word_counter = 0

    for image_filename in tqdm(image_files, desc="Extracting Words"):
        image_base_name = os.path.splitext(image_filename)[0]
        
        # Possible bounding box file patterns
        possible_bb_names = [
            f"res_{image_base_name}_sorted.txt",
            f"res_{image_base_name.replace('image_', 'page_')}_sorted.txt",
            f"res_img_{image_base_name}_sorted.txt"
        ]
        
        bounding_box_path = None
        for bb_name in possible_bb_names:
            bb_path = os.path.join(bounding_box_folder, bb_name)
            if os.path.exists(bb_path):
                bounding_box_path = bb_path
                break

        if bounding_box_path:
            image_path = os.path.join(image_folder, image_filename)
            log.debug(f"Processing {image_filename} with bounding boxes from {bounding_box_path}")
            word_counter = extract_bounding_boxes(image_path, bounding_box_path, output_folder, word_counter)
        else:
            log.warning(f"Bounding box file for {image_filename} not found. Tried: {possible_bb_names}")
            
    log.info(f"Total words extracted: {word_counter}")
    return word_counter

def pad_and_resize_images(folder_path: str, target_size: Tuple[int, int] = IMAGE_SIZE) -> None:
    """
    Resizes images in a folder to target_size, padding to maintain aspect ratio.
    
    Args:
        folder_path: Path to folder containing images
        target_size: Target size as (width, height)
    """
    log.info(f"Padding and resizing images in {folder_path} to {target_size}...")
    target_width, target_height = target_size
    
    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    if not image_files:
        log.warning(f"No PNG images found in {folder_path}")
        return

    successful = 0
    failed = 0

    for img_path in tqdm(image_files, desc="Padding/Resizing"):
        try:
            with Image.open(img_path) as img:
                if img.mode != 'L':
                    img = img.convert("L")
                
                original_width, original_height = img.size
                
                if original_width <= 0 or original_height <= 0:
                    log.warning(f"Invalid image dimensions in {img_path}")
                    continue
                
                # Calculate scaling ratio to maintain aspect ratio
                ratio = min(target_width / original_width, target_height / original_height)
                if ratio <= 0:
                    ratio = 1
                
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                
                # Resize with high-quality interpolation
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Create new white background image
                new_img = Image.new("L", target_size, 255)
                
                # Calculate paste position to center the image
                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                new_img.paste(img_resized, (paste_x, paste_y))
                
                # Save with optimal compression
                new_img.save(img_path, optimize=True, quality=95)
                successful += 1
                
        except Exception as e:
            log.warning(f"Error processing {img_path}: {e}")
            failed += 1
    
    log.info(f"Padding/Resizing complete. Successful: {successful}, Failed: {failed}")

# --- Augmentation Functions (for training) ---

def rotation_aug(folder_path: str, angles: List[int] = [-5, -3, -1, 1, 3, 5]) -> None:
    """
    Applies rotation augmentation to images.
    
    Args:
        folder_path: Path to folder containing images
        angles: List of rotation angles in degrees
    """
    log.info(f"Applying rotation augmentation in {folder_path} with angles {angles}...")
    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    
    successful = 0
    skipped = 0
    
    for img_path in tqdm(image_files, desc="Augment: Rotate"):
        # Skip already augmented files
        if any(f"_rot{angle}" in img_path for angle in angles):
            skipped += 1
            continue
            
        try:
            with Image.open(img_path) as img:
                if img.mode != 'L':
                    img = img.convert("L")
                
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                dir_name = os.path.dirname(img_path)
                
                for angle in angles:
                    # Rotate around center
                    rotated_img = img.rotate(
                        angle, 
                        resample=Image.BICUBIC, 
                        expand=False, 
                        fillcolor=255,  # White background
                        center=(img.width // 2, img.height // 2)
                    )
                    aug_name = f"{base_name}_rot{angle}.png"
                    rotated_img.save(os.path.join(dir_name, aug_name), optimize=True)
                    successful += 1
                    
        except Exception as e:
            log.warning(f"Error rotating {img_path}: {e}")
    
    log.info(f"Rotation augmentation complete. Created: {successful}, Skipped: {skipped}")

def gaussian_noise_aug(folder_path: str, mean: float = 0, var: float = 0.01) -> None:
    """
    Applies gaussian noise augmentation to images.
    
    Args:
        folder_path: Path to folder containing images
        mean: Mean of Gaussian distribution
        var: Variance of Gaussian distribution
    """
    log.info(f"Applying Gaussian noise augmentation in {folder_path} (mean={mean}, var={var})...")
    image_files = glob.glob(os.path.join(folder_path, "*.png")) 
    
    successful = 0
    skipped = 0
    
    for img_path in tqdm(image_files, desc="Augment: Noise"):
        if "_gauss" in img_path:
            skipped += 1
            continue
            
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            # Calculate standard deviation
            sigma = math.sqrt(var)
            
            # Generate Gaussian noise
            gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
            
            # Add noise and clip to valid range
            noisy_img = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            dir_name = os.path.dirname(img_path)
            aug_name = f"{base_name}_gauss.png"
            cv2.imwrite(os.path.join(dir_name, aug_name), noisy_img)
            successful += 1
            
        except Exception as e:
            log.warning(f"Error adding noise to {img_path}: {e}")
    
    log.info(f"Gaussian noise augmentation complete. Created: {successful}, Skipped: {skipped}")

def create_csv_from_folder(folder_path: str, csv_path: str) -> None:
    """
    Creates a CSV file listing all .png files and their identity.
    
    Args:
        folder_path: Path to folder containing images
        csv_path: Output CSV file path
    """
    log.info(f"Creating CSV file: {csv_path}")
    files = glob.glob(os.path.join(folder_path, "*.png"))
    files = [f for f in files if os.path.basename(f) != ".png" and "_rot" not in f and "_gauss" not in f]
    
    if not files:
        log.warning(f"No valid PNG files found in {folder_path}")
        return
        
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["FILENAME", "IDENTITY"])
        for file_path in files:
            filename = os.path.basename(file_path)
            identity = os.path.splitext(filename)[0]
            writer.writerow([filename, identity])
            
    log.info(f'CSV file "{csv_path}" created with {len(files)} entries.')