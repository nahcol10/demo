import os
import logging
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Constants ---
IMG_WIDTH = 200
IMG_HEIGHT = 50
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# --- TensorFlow Model Components (kept for compatibility) ---
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
        decoded_image = tf.image.decode_png(contents=image, channels=1)
        cnvt_image = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
        resized_image = tf.image.resize(images=cnvt_image, size=(IMG_HEIGHT, IMG_WIDTH))
        image = tf.transpose(resized_image, perm=[1, 0, 2])
        image = tf.cast(image, dtype=tf.float32)
        return image
    except Exception as e:
        log.error(f"Error loading image {image_path}: {e}")
        # Return a blank image as fallback
        return tf.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32)

# --- Simple Image Processing Utilities ---
def pad_and_resize_image(image_path: str, target_size: tuple = IMAGE_SIZE) -> None:
    """
    Resizes an image to target_size, padding to maintain aspect ratio.
    Args:
        image_path: Path to image file
        target_size: Target size as (width, height)
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'L':
                img = img.convert("L")
            original_width, original_height = img.size
            if original_width <= 0 or original_height <= 0:
                log.warning(f"Invalid image dimensions in {image_path}")
                return
                
            target_width, target_height = target_size
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
            new_img.save(image_path, optimize=True, quality=95)
    except Exception as e:
        log.warning(f"Error processing {image_path}: {e}")

# --- Vocabulary Management (kept for potential future use) ---
def save_vocabulary(path: str, unique_chars: list, max_label_length: int) -> None:
    """
    Saves the character list and max length to a JSON file.
    Args:
        path: Output file path
        unique_chars: List of unique characters
        max_label_length: Maximum label length
    """
    vocab_data = {
        'vocabulary': list(unique_chars), 
        'max_label_length': int(max_label_length)
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    log.info(f"Vocabulary saved to {path} - {len(unique_chars)} characters, max length: {max_label_length}")

def load_vocabulary(path: str) -> tuple:
    """
    Loads vocabulary and model parameters from JSON file.
    Args:
        path: Path to vocabulary JSON file
    Returns:
        Tuple of (char_to_num, num_to_char, n_classes, max_label_length)
    """
    if not os.path.exists(path):
        log.error(f"Vocabulary file not found: {path}")
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        unique_chars = vocab_data['vocabulary']
        max_label_length = vocab_data['max_label_length']
        n_classes = len(unique_chars)
        
        # Create lookup layers
        char_to_num = layers.StringLookup(vocabulary=unique_chars, mask_token=None, oov_token='[UNK]', name='char_to_num')
        num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True, name='num_to_char')
        
        log.info(f"Vocabulary loaded: {n_classes} unique characters, max length: {max_label_length}")
        return char_to_num, num_to_char, n_classes, max_label_length
    except Exception as e:
        log.error(f"Error loading vocabulary from {path}: {e}")
        raise