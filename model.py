import os
import csv
import json
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
from tqdm import tqdm

log = logging.getLogger(__name__)

# --- Constants ---
IMG_WIDTH = 200
IMG_HEIGHT = 50

# ===================================================================
# DATA PREPARATION FUNCTIONS (from your old utils.py)
# ===================================================================

def pad_and_resize_images(folder_path, target_size=(IMG_WIDTH, IMG_HEIGHT)):
    """Pads and resizes all images in a folder."""
    log.info(f"Padding and resizing images in {folder_path}...")
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Resizing images"):
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('L') # Open as grayscale
            
            # Pad to a 4:1 aspect ratio (width:height)
            w, h = img.size
            target_w = h * 4
            if w < target_w:
                padding = (target_w - w) // 2
                img = ImageOps.pad(img, (target_w, h), color='white', centering=(0.5, 0.5))
            
            # Resize to target size
            img = img.resize(target_size, Image.LANCZOS)
            
            img.save(img_path)
        except Exception as e:
            log.warning(f"Could not process image {filename}: {e}")

def rotation_aug(folder_path, angles=[-5, 5]):
    """Applies rotation augmentation to images."""
    log.info(f"Applying rotation augmentation in {folder_path}...")
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="Augmenting (Rotation)"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        for angle in angles:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            
            new_filename = f"{os.path.splitext(filename)[0]}_rot{angle}.png"
            cv2.imwrite(os.path.join(folder_path, new_filename), rotated)

def gaussian_noise_aug(folder_path, mean=0, var=0.1):
    """Applies gaussian noise augmentation."""
    log.info(f"Applying gaussian noise augmentation in {folder_path}...")
    # Process original and rotated images
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg')) and '_gauss' not in f]
    
    for filename in tqdm(image_files, desc="Augmenting (Noise)"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale
        if img is None:
            continue

        img = img / 255.0 # Normalize
        noise = np.random.normal(mean, var**0.5, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0.0, 1.0)
        noisy_img = (255 * noisy_img).astype(np.uint8)
        
        new_filename = f"{os.path.splitext(filename)[0]}_gauss.png"
        cv2.imwrite(os.path.join(folder_path, new_filename), noisy_img)

def create_csv_from_folder(folder_path, csv_path):
    """Creates a CSV file from image filenames."""
    log.info(f"Creating CSV file at {csv_path}...")
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['FILENAME', 'IDENTITY'])
        
        for filename in image_files:
            # Label is the filename before any underscores (from augmentation)
            label = filename.split('_')[0]
            writer.writerow([filename, label])

def save_vocabulary(filepath, vocab, max_length):
    """Saves the vocabulary and max length to a JSON file."""
    log.info(f"Saving vocabulary to {filepath}...")
    with open(filepath, 'w') as f:
        json.dump({
            'vocabulary': vocab,
            'max_label_length': max_length
        }, f, indent=2)

# ===================================================================
# TENSORFLOW & MODEL FUNCTIONS (from your Model.ipynb)
# ===================================================================

class CTCLayer(layers.Layer):
    """Custom CTC Loss Layer (from Model.ipynb Cell 45)"""
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

    def get_config(self):
        config = super().get_config()
        return config

def load_image(image_path: str):
    """Loads and preprocesses an image (from Model.ipynb Cell 37)"""
    image = tf.io.read_file(image_path)
    decoded_image = tf.image.decode_jpeg(contents=image, channels=1)
    cnvt_image = tf.image.convert_image_dtype(image=decoded_image, dtype=tf.float32)
    resized_image = tf.image.resize(images=cnvt_image, size=(IMG_HEIGHT, IMG_WIDTH))
    image = tf.transpose(resized_image, perm=[1, 0, 2])
    image = tf.cast(image, dtype=tf.float32)
    return image

def encode_single_sample(image_path: str, label: str, max_label_length: int, 
                         char_to_num: layers.StringLookup, vocab_size: int):
    """Encodes a single sample for tf.data (from Model.ipynb Cell 38)"""
    image = load_image(image_path)
    chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
    vecs = char_to_num(chars)
    
    pad_size = max_label_length - tf.shape(vecs)[0]
    vecs = tf.pad(vecs, paddings=[[0, pad_size]], constant_values=vocab_size) # Use vocab_size for padding
    
    return {'image': image, 'label': vecs}

def build_ocr_model(vocab_size: int, img_width=IMG_WIDTH, img_height=IMG_HEIGHT) -> tuple[keras.Model, keras.Model]:
    """
    Builds the Keras OCR model (from Model.ipynb Cell 47)
    
    Returns:
        (model, inference_model): A tuple of the training model and the inference-only model
    """
    
    # Inputs
    input_images = layers.Input(shape=(img_width, img_height, 1), name="image")
    target_labels = layers.Input(shape=(None,), name="label")

    # CNN Network
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input_images)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    # Encoding Space
    encoding = layers.Reshape(target_shape=((img_width // 4), (img_height // 4) * 128))(x)
    encoding = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(encoding)
    encoding = layers.Dropout(0.5)(encoding)

    # RNN Network
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.5))(encoding)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.5))(x)

    # Output Layer
    # Use vocab_size + 1 for the CTC blank token
    output = layers.Dense(vocab_size + 1, activation='softmax', name='output_dense')(x)

    # CTC Layer
    ctc_layer = CTCLayer()(target_labels, output)

    # Model
    model = keras.Model(
        inputs=[input_images, target_labels],
        outputs=[ctc_layer],
        name="training_model"
    )
    
    # --- CREATE INFERENCE MODEL ---
    inference_model = keras.Model(
        inputs=input_images,
        outputs=output,
        name="inference_model"
    )
    
    return model, inference_model

def decode_pred(pred_label, num_to_char: layers.StringLookup):
    """Decodes model predictions (from Model.ipynb Cell 46)"""
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]
    
    # Use a large max_label_length for decoding, or get from model metadata if possible
    max_label_length = 100 
    
    decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][:, :max_label_length]
    
    chars = num_to_char(decode)
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]
    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]
    return filtered_texts