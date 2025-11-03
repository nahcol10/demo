#!/usr/bin/env python
# coding: utf-8

# --- IMPORTS ---
import os
import logging
import cv2
import numpy as np
import json
import math
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- CONSTANTS ---
IMG_WIDTH = 200
IMG_HEIGHT = 50
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# --- Custom CTC Layer ---

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

# --- Model Building Function ---

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

# --- Data Loading and Processing Functions ---

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
        'label_length': final_len_expanded
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

# --- Image Processing Utilities ---

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

def pad_and_resize_images(directory: str, target_size: tuple = IMAGE_SIZE) -> None:
    """
    Processes all images in a directory, padding and resizing them.
    
    Args:
        directory: Path to directory containing images
        target_size: Target size as (width, height)
    """
    if not os.path.exists(directory):
        log.error(f"Directory not found: {directory}")
        return
    
    import glob
    image_files = glob.glob(os.path.join(directory, "*.png"))
    log.info(f"Processing {len(image_files)} images in {directory}")
    
    for image_path in image_files:
        pad_and_resize_image(image_path, target_size)
    
    log.info(f"Completed padding and resizing {len(image_files)} images")

# --- Data Augmentation Functions ---

def rotation_aug(directory: str, angles: List[int] = None) -> None:
    """
    Applies rotation augmentation to images in a directory.
    
    Args:
        directory: Path to directory containing images
        angles: List of rotation angles in degrees
    """
    if angles is None:
        angles = [-5, 5]
    
    if not os.path.exists(directory):
        log.error(f"Directory not found: {directory}")
        return
    
    import glob
    image_files = glob.glob(os.path.join(directory, "*.png"))
    log.info(f"Applying rotation augmentation to {len(image_files)} images")
    
    count = 0
    for image_path in image_files:
        # Skip already augmented images
        if any(f"_rot{angle}" in image_path for angle in angles):
            continue
            
        try:
            with Image.open(image_path) as img:
                if img.mode != 'L':
                    img = img.convert("L")
                
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                dir_name = os.path.dirname(image_path)
                
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
                    count += 1
                    
        except Exception as e:
            log.warning(f"Error rotating {image_path}: {e}")
    
    log.info(f"Rotation augmentation complete. Created: {count} images")

def gaussian_noise_aug(directory: str, mean: float = 0, sigma: float = 5) -> None:
    """
    Applies Gaussian noise augmentation to images in a directory.
    
    Args:
        directory: Path to directory containing images
        mean: Mean of the Gaussian noise
        sigma: Standard deviation of the Gaussian noise
    """
    if not os.path.exists(directory):
        log.error(f"Directory not found: {directory}")
        return
    
    import glob
    image_files = glob.glob(os.path.join(directory, "*.png"))
    log.info(f"Applying Gaussian noise augmentation to {len(image_files)} images")
    
    count = 0
    for image_path in image_files:
        # Skip already augmented images
        if '_gauss' in image_path or '_noise' in image_path:
            continue
            
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Generate Gaussian noise
            noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
            
            # Add noise to image
            noisy_img = img.astype(np.float32) + noise
            
            # Clip values to valid range
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            # Save augmented image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            dir_name = os.path.dirname(image_path)
            aug_name = f"{base_name}_gauss.png"
            cv2.imwrite(os.path.join(dir_name, aug_name), noisy_img)
            count += 1
        except Exception as e:
            log.warning(f"Error applying noise to {image_path}: {e}")
    
    log.info(f"Gaussian noise augmentation complete. Created: {count} images")

# --- CSV Creation ---

def create_csv_from_folder(directory: str, output_path: str) -> None:
    """
    Creates a CSV file from images in a directory.
    Assumes filenames follow the pattern: label_*.png
    
    Args:
        directory: Path to directory containing images
        output_path: Path to save the CSV file
    """
    import pandas as pd
    import glob
    
    if not os.path.exists(directory):
        log.error(f"Directory not found: {directory}")
        return
    
    image_files = glob.glob(os.path.join(directory, "*.png"))
    
    data = []
    for image_path in image_files:
        image_file = os.path.basename(image_path)
        # Extract label from filename (assuming format: label_*.ext)
        base_name = os.path.splitext(image_file)[0]
        
        # Handle augmented filenames
        if '_rot' in base_name:
            label = base_name.split('_rot')[0]
        elif '_gauss' in base_name:
            label = base_name.split('_gauss')[0]
        elif '_noise' in base_name:
            label = base_name.split('_noise')[0]
        else:
            # Try to extract label before first underscore or use the whole name
            parts = base_name.split('_')
            if len(parts) > 0:
                label = parts[0]
            else:
                label = base_name
        
        data.append({
            'FILENAME': image_file,
            'IDENTITY': label
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    log.info(f"Created CSV with {len(df)} entries at {output_path}")

# --- Simple Encoding Function for Training ---

def encode_single_sample_simple(img_path: str, label: str, max_label_length: int, 
                               char_to_num: layers.StringLookup, vocab_size: int) -> dict:
    """
    Simplified encoding function for training data.
    
    Args:
        img_path: Path to image file
        label: Text label
        max_label_length: Maximum label length
        char_to_num: Character to number lookup layer
        vocab_size: Size of vocabulary
    
    Returns:
        Dictionary with 'image' and 'label' tensors
    """
    # Load and process image
    img = load_image(img_path)
    
    # Convert label to character indices
    label_chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
    label_indices = char_to_num(label_chars)
    
    # Pad label to max length
    label_indices = tf.pad(label_indices, [[0, max_label_length - tf.shape(label_indices)[0]]], 
                          constant_values=0)  # Pad with [UNK] token
    
    return {"image": img, "label": label_indices}