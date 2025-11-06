import os
import fitz  # PyMuPDF
import cv2
import easyocr
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

IMAGE_SIZE = (32, 128)  # Height, Width

# --- 1. Data Generation ---
# ... (all functions from pdf_to_images to create_dataset_from_pdf remain unchanged) ...
def pdf_to_images(pdf_path, output_folder, dpi=300):
    """Converts each page of a PDF to a high-resolution image."""
    log.info(f"Converting PDF '{pdf_path}' to images...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in tqdm(range(len(doc)), desc="Converting PDF pages"):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(output_folder, f"page_{page_num}.png")
        pix.save(img_path)
        image_paths.append(img_path)
        
    doc.close()
    log.info(f"Saved {len(image_paths)} images to '{output_folder}'.")
    return image_paths

def extract_bounding_boxes_from_image(img_path, reader):
    """Uses EasyOCR to find text bounding boxes in a single image."""
    try:
        image = cv2.imread(img_path)
        if image is None:
            log.warning(f"Could not read image: {img_path}")
            return [], None
            
        results = reader.readtext(image, detail=1)
        
        # Sort results top-to-bottom, left-to-right
        def get_sort_key(res):
            bbox = res[0]
            top_left = bbox[0]
            return (top_left[1], top_left[0]) # y, then x

        sorted_results = sorted(results, key=get_sort_key)
        return sorted_results, image
        
    except Exception as e:
        log.error(f"Error processing image {img_path} with EasyOCR: {e}")
        return [], None

def create_dataset_from_pdf(pdf_path, output_dir, lang_list=['ne', 'en']):
    """
    Full pipeline:
    1. Converts PDF to images.
    2. Uses EasyOCR to detect text in images.
    3. Crops word images and saves them to output_dir.
    """
    log.info("Initializing EasyOCR reader...")
    reader = easyocr.Reader(lang_list)
    
    temp_image_folder = os.path.join(output_dir, "temp_pages")
    word_output_dir = os.path.join(output_dir, "words")
    
    if not os.path.exists(word_output_dir):
        os.makedirs(word_output_dir)
        
    # 1. Convert PDF to images
    image_paths = pdf_to_images(pdf_path, temp_image_folder)
    
    # 2. Extract words from each image
    log.info("Extracting words from page images...")
    total_words = 0
    for page_num, img_path in enumerate(tqdm(image_paths, desc="Processing pages")):
        sorted_results, image = extract_bounding_boxes_from_image(img_path, reader)
        
        if image is None:
            continue
            
        for i, (bbox, text, prob) in enumerate(sorted_results):
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            
            # Add some padding
            y_min = max(0, tl[1] - 5)
            y_max = min(image.shape[0], br[1] + 5)
            x_min = max(0, tl[0] - 5)
            x_max = min(image.shape[1], br[0] + 5)
            
            cropped_word = image[y_min:y_max, x_min:x_max]
            
            if cropped_word.size == 0:
                log.warning(f"Empty crop at page {page_num}, word {i}. Skipping.")
                continue
            
            # Save cropped word
            word_filename = f"word_{total_words}_page_{page_num}.png"
            cv2.imwrite(os.path.join(word_output_dir, word_filename), cropped_word)
            total_words += 1
            
    log.info(f"Successfully extracted {total_words} words to '{word_output_dir}'.")
    return word_output_dir

# --- 2. Model & Training ---

def load_and_preprocess_image(img_path, image_size=IMAGE_SIZE, channels=1):
    """Loads and preprocesses a single image for the model."""
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    target_h, target_w = image_size
    
    # Resize keeping aspect ratio
    scale = tf.minimum(target_w / w, target_h / h)
    new_w, new_h = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32), tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    
    image = tf.image.resize(image, [new_h, new_w], method='area')
    
    # Pad to target size
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    image = tf.pad(image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=1.0)
    
    # Transpose for CTC (Width, Height, Channels)
    image = tf.transpose(image, perm=[1, 0, 2])
    return image

class CTCLayer(layers.Layer):
    """CTC (Connectionist Temporal Classification) layer."""
    
    def __init__(self, name="ctc_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    
    def call(self, y_true, y_pred, input_length, label_length):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        
        # Setup loss calculation
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
    
    def get_config(self):
        config = super().get_config()
        return config

def build_model(input_shape, n_classes):
    """Builds the CRNN (CNN + RNN) model."""
    inputs = layers.Input(shape=input_shape, name="image")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
    label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")

    # --- CNN ---
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPool2D((2, 2))(x)
    
    # --- Reshape for RNN ---
    # (Width, Height, Channels) -> (Width, Height * Channels)
    # Input shape to RNN is (batch_size, timesteps, features)
    # Here, timesteps = Width // 4, features = (Height // 4) * 64
    new_shape = (input_shape[0] // 4, (input_shape[1] // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.2)(x)
    
    # --- RNN ---
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    
    # --- Output ---
    x = layers.Dense(n_classes + 1, activation="softmax", name="softmax_output")(x)
    
    # --- CTC Loss ---
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)
    
    # --- Model Definition ---
    model = keras.models.Model(
        inputs=[inputs, labels, input_length, label_length],
        outputs=output,
        name="ocr_model_v1"
    )
    
    # --- Inference Model ---
    # Create a separate model for prediction
    inference_model = keras.models.Model(
        inputs=inputs,
        outputs=x,
        name="inference_model"
    )
    
    # === REMOVED THE FAULTY LINE ===
    # model.add_layer(inference_model)
    
    return model, inference_model # === RETURN BOTH MODELS ===

# --- 3. Evaluation ---

def decode_batch_predictions(pred, num_to_char, max_label_length):
    """Decodes a batch of predictions."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_label_length
    ]
    
    output_texts = []
    for res in results:
        res = tf.gather(res, tf.where(tf.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_texts.append(res)
    return output_texts

def calculate_cer(ground_truth, predicted):
    """Calculates Character Error Rate (CER)."""
    matcher = SequenceMatcher(None, ground_truth, predicted)
    num_edits = sum(triple[-1] for triple in matcher.get_opcodes() if triple[0] != 'equal')
    return num_edits / len(ground_truth) if len(ground_truth) > 0 else 1.0

# === CHANGED `model` to `inference_model` ===
def evaluate_model(inference_model, valid_df, num_to_char, image_dir='data/words', batch_size=64):
    """Evaluates the model on the validation set."""
    log.info("Evaluating model...")
    
    # === NO LONGER NEEDED ===
    # try:
    #     inference_model = model.get_layer("inference_model")
    # except ValueError:
    #     log.error("Could not find 'inference_model' layer. Was the model built with utils.build_model?")
    #     return

    # Create a simple generator
    images = []
    for _, row in valid_df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        images.append(load_and_preprocess_image(img_path, image_size=IMAGE_SIZE))
    
    image_batch = tf.stack(images, axis=0)
    
    # Predict
    preds = inference_model.predict(image_batch)
    
    # Decode
    pred_texts = decode_batch_predictions(preds, num_to_char, valid_df['label'].str.len().max())
    true_texts = valid_df['label'].tolist()
    
    # Calculate CER
    cers = [calculate_cer(true, pred) for true, pred in zip(true_texts, pred_texts)]
    avg_cer = np.mean(cers)
    
    # Calculate Accuracy
    accuracy = sum(1 for true, pred in zip(true_texts, pred_texts) if true == pred) / len(true_texts)
    
    log.info(f"--- Evaluation Results ---")
    log.info(f"Average Character Error Rate (CER): {avg_cer:.4f}")
    log.info(f"Exact Match Accuracy: {accuracy:.4f}")
    
    # Print some examples
    log.info("--- Examples ---")
    for i in range(min(10, len(true_texts))):
        log.info(f"True: '{true_texts[i]}' | Pred: '{pred_texts[i]}'")
        
    return avg_cer, accuracy

def plot_training_history(history):
    """Plots the training and validation loss."""
    log.info("Plotting training history...")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join('plots', 'training_history.png')
    plt.savefig(plot_path)
    log.info(f"Saved training plot to {plot_path}")