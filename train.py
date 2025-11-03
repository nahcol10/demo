import os
import sys
import shutil
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import gdown
import logging
from typing import Tuple, Dict, Any
import json

import model as ocr_model  # Imports from model.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- CONSTANTS ---
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = 'ocr_model.h5'
VOCAB_SAVE_PATH = 'vocabulary.json'
CHECKPOINT_PATH = 'model_checkpoints/epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'

# Data paths
BASE_DIR = '.'
TRAINING_ZIP_PATH = os.path.join(BASE_DIR, 'outputCorrected.zip')
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')
TRAINING_CSV_PATH = os.path.join(BASE_DIR, 'training_data.csv')
TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'train.csv')
VALID_CSV_PATH = os.path.join(BASE_DIR, 'valid.csv')
PLOTS_DIR = 'training_plots'

def download_training_data() -> None:
    """Downloads and extracts training data if not already present."""
    if not os.path.exists(TRAINING_DATA_DIR):
        if not os.path.exists(TRAINING_ZIP_PATH):
            log.info("Downloading corrected training data...")
            try:
                gdown.download(
                    id='1t3CoVe2m6kUQJ7VbiK3xZX8sDV69qxPW', 
                    output=TRAINING_ZIP_PATH,
                    quiet=False
                )
            except Exception as e:
                log.error(f"Failed to download training data: {e}")
                raise
        
        log.info("Unzipping training data...")
        try:
            with zipfile.ZipFile(TRAINING_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            
            # Handle different possible folder structures
            possible_folders = ['labeled_data', 'training_data', 'data']
            for folder in possible_folders:
                if os.path.exists(folder):
                    if os.path.exists(TRAINING_DATA_DIR):
                        shutil.rmtree(TRAINING_DATA_DIR)
                    shutil.move(folder, TRAINING_DATA_DIR)
                    break
            
            if not os.path.exists(TRAINING_DATA_DIR):
                log.error("Could not find training data folder after extraction")
                raise FileNotFoundError("Training data folder not found")
            
            # Clean up zip file
            os.remove(TRAINING_ZIP_PATH)
            log.info(f"Training data extracted to {TRAINING_DATA_DIR}")
            
        except Exception as e:
            log.error(f"Error extracting training data: {e}")
            raise
    else:
        log.info("Training data directory already exists.")

def prepare_training_data() -> None:
    """Preprocesses and augments training data."""
    log.info("Preparing training data...")
    
    # Pad and resize images
    ocr_model.pad_and_resize_images(TRAINING_DATA_DIR)
    
    # Apply augmentations
    ocr_model.rotation_aug(TRAINING_DATA_DIR)
    ocr_model.gaussian_noise_aug(TRAINING_DATA_DIR)
    
    log.info("Training data preparation complete.")

def create_training_csvs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates training and validation CSV files.
    
    Returns:
        Tuple of (train_df, valid_df)
    """
    log.info("Creating training CSVs...")
    
    # Create base CSV
    ocr_model.create_csv_from_folder(TRAINING_DATA_DIR, TRAINING_CSV_PATH)
    
    # Load and clean the data
    df = pd.read_csv(TRAINING_CSV_PATH)
    
    # Clean IDENTITY from augmentation suffixes and ensure string type
    df['IDENTITY'] = df['IDENTITY'].apply(
        lambda x: str(x).split('_')[0] if isinstance(x, str) else str(x)
    ).astype(str)
    
    # Remove empty or invalid labels
    df = df[df['IDENTITY'].str.len() > 0]
    df = df[df['IDENTITY'] != 'nan']
    df = df[df['IDENTITY'].str.strip() != '']
    
    # Shuffle with fixed seed for reproducibility
    df = df.sample(frac=1, random_state=2569).reset_index(drop=True)
    
    # Split into train and validation sets
    TRAIN_SIZE = int(len(df) * 0.8)
    df_train = df.iloc[:TRAIN_SIZE].copy()
    df_valid = df.iloc[TRAIN_SIZE:].copy()
    
    # Save the splits
    df_train.to_csv(TRAIN_CSV_PATH, index=False)
    df_valid.to_csv(VALID_CSV_PATH, index=False)
    
    log.info(f"Training CSVs created. Train: {len(df_train)}, Valid: {len(df_valid)}")
    log.info(f"Sample training labels: {df_train['IDENTITY'].head().tolist()}")
    
    return df_train, df_valid

def prepare_vocabulary(df: pd.DataFrame) -> Tuple[layers.StringLookup, layers.StringLookup, int, int]:
    """
    Prepares vocabulary from training data.
    
    Args:
        df: DataFrame containing training data
    
    Returns:
        Tuple of (char_to_num, num_to_char, n_classes, max_label_length)
    """
    log.info("Preparing vocabulary...")
    
    labels = [str(word).strip() for word in df['IDENTITY'].to_numpy() if str(word).strip()]
    
    if not labels:
        raise ValueError("No valid labels found in training data")
    
    # Get all unique characters
    all_chars = []
    for word in labels:
        all_chars.extend(list(word))
    
    unique_chars = sorted(list(set(all_chars)))
    n_classes = len(unique_chars)
    max_label_length = max(len(word) for word in labels)
    
    log.info(f"Total unique characters: {n_classes}")
    log.info(f"Max label length: {max_label_length}")
    log.info(f"Unique characters: {''.join(unique_chars)}")
    
    # Save vocabulary for inference
    ocr_model.save_vocabulary(VOCAB_SAVE_PATH, unique_chars, max_label_length)
    
    # Create lookup layers
    char_to_num = layers.StringLookup(
        vocabulary=unique_chars, 
        mask_token=None, 
        oov_token='[UNK]',
        name='char_to_num'
    )
    
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), 
        mask_token=None, 
        invert=True,
        name='num_to_char'
    )
    
    return char_to_num, num_to_char, n_classes, max_label_length

def create_datasets(train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                   char_to_num: layers.StringLookup, max_label_length: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creates TensorFlow datasets for training and validation.
    
    Args:
        train_df: Training DataFrame
        valid_df: Validation DataFrame
        char_to_num: Character to number lookup layer
        max_label_length: Maximum label length
    
    Returns:
        Tuple of (train_ds, valid_ds)
    """
    log.info("Creating TensorFlow datasets...")
    
    # Add full paths to filenames
    train_df['FILENAME'] = train_df['FILENAME'].apply(lambda x: os.path.join(TRAINING_DATA_DIR, x))
    valid_df['FILENAME'] = valid_df['FILENAME'].apply(lambda x: os.path.join(TRAINING_DATA_DIR, x))
    
    # Verify files exist
    train_files = train_df['FILENAME'].tolist()
    valid_files = valid_df['FILENAME'].tolist()
    
    missing_train = [f for f in train_files if not os.path.exists(f)]
    missing_valid = [f for f in valid_files if not os.path.exists(f)]
    
    if missing_train:
        log.warning(f"Missing {len(missing_train)} training files, removing from dataset")
        train_df = train_df[~train_df['FILENAME'].isin(missing_train)]
    
    if missing_valid:
        log.warning(f"Missing {len(missing_valid)} validation files, removing from dataset")
        valid_df = valid_df[~valid_df['FILENAME'].isin(missing_valid)]
    
    log.info(f"Final dataset sizes - Train: {len(train_df)}, Valid: {len(valid_df)}")
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # --- START FIX ---
    # Calculate vocab_size *outside* the mapped function (in Eager mode).
    # This value is constant and can't be computed inside the graph.
    vocab_size = len(char_to_num.get_vocabulary())
    
    # Create encoding function with required parameters
    def encode_train(img_path, label):
        return ocr_model.encode_single_sample(
            img_path, 
            label, 
            max_label_length, 
            char_to_num, 
            vocab_size  # Pass the pre-calculated integer
        )
    # --- END FIX ---
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_df['FILENAME'].values, train_df['IDENTITY'].values)
    )
    train_ds = train_ds.shuffle(len(train_df), seed=2569)
    train_ds = train_ds.map(encode_train, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    
    valid_ds = tf.data.Dataset.from_tensor_slices(
        (valid_df['FILENAME'].values, valid_df['IDENTITY'].values)
    )
    valid_ds = valid_ds.map(encode_train, num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE)
    valid_ds = valid_ds.prefetch(AUTOTUNE)
    
    # Test dataset creation
    try:
        for batch in train_ds.take(1):
            log.info("Training dataset batch structure:")
            for key, value in batch.items():
                log.info(f"  {key}: {value.shape}")
        log.info("Dataset creation successful")
    except Exception as e:
        log.error(f"Error creating datasets: {e}")
        raise
    
    return train_ds, valid_ds

def build_and_train_model(train_ds: tf.data.Dataset, valid_ds: tf.data.Dataset, 
                         n_classes: int, char_to_num: layers.StringLookup) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Builds, compiles, and trains the OCR model.
    
    Args:
        train_ds: Training dataset
        valid_ds: Validation dataset
        n_classes: Number of character classes
        char_to_num: Character to number lookup layer
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    log.info("Building model...")
    
    # Get vocabulary size including [UNK] token
    vocab_size = len(char_to_num.get_vocabulary())
    log.info(f"Vocabulary size: {vocab_size}")
    
    model = ocr_model.build_ocr_model(n_classes, vocab_size)
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=None  # Loss is handled by the CTCLayer
    )
    
    # Create callbacks
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            verbose=1, 
            min_lr=1e-6
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1,
            write_images=True
        )
    ]
    
    log.info("Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history

def plot_training_history(history: Dict[str, Any]) -> None:
    """
    Plots and saves training history.
    
    Args:
        history: Training history dictionary
    """
    log.info("Plotting training history...")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(PLOTS_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"Training history plot saved to {plot_path}")

def evaluate_model(model: keras.Model, valid_df: pd.DataFrame, 
                  num_to_char: layers.StringLookup) -> None:
    """
    Evaluates the trained model on validation data.
    
    Args:
        model: Trained Keras model
        valid_df: Validation DataFrame
        num_to_char: Number to character lookup layer
    """
    log.info("Evaluating model on validation data...")
    
    # Take a small sample for evaluation
    sample_size = min(10, len(valid_df))
    sample_df = valid_df.sample(sample_size, random_state=42)
    
    predictions = []
    ground_truth = []
    
    for _, row in sample_df.iterrows():
        try:
            img_path = row['FILENAME']
            label = row['IDENTITY']
            
            # Load and preprocess image
            img = ocr_model.load_image(img_path)
            img = tf.expand_dims(img, 0)  # Add batch dimension
            
            # Get prediction
            pred = model.predict(img, verbose=0)
            decoded = ocr_model.decode_pred(pred, num_to_char)
            
            predictions.append(decoded[0])
            ground_truth.append(label)
            
        except Exception as e:
            log.warning(f"Error evaluating sample {img_path}: {e}")
    
    # Print evaluation results
    log.info("\nValidation Sample Results:")
    log.info("-" * 50)
    for pred, gt in zip(predictions, ground_truth):
        log.info(f"Predicted: '{pred}'")
        log.info(f"Ground Truth: '{gt}'")
        log.info("-" * 30)
    
    # Calculate simple accuracy (exact match)
    exact_matches = sum(1 for p, g in zip(predictions, ground_truth) if p.strip() == g.strip())
    accuracy = exact_matches / len(predictions) if predictions else 0
    
    log.info(f"\nExact Match Accuracy: {accuracy:.2%} ({exact_matches}/{len(predictions)})")

def main():
    """Main training pipeline."""
    log.info("=" * 60)
    log.info("Starting OCR Model Training Pipeline")
    log.info("=" * 60)
    
    try:
        # Set Random Seeds
        np.random.seed(2569)
        tf.random.set_seed(2569)
        
        # --- 1. Download & Prepare Training Data ---
        download_training_data()
        
        # --- 2. Preprocess & Augment Training Data ---
        prepare_training_data()
        
        # --- 3. Create CSVs ---
        train_df, valid_df = create_training_csvs()
        
        # --- 4. Prepare Vocabulary & Datasets ---
        char_to_num, num_to_char, n_classes, max_label_length = prepare_vocabulary(pd.concat([train_df, valid_df]))
        
        # --- 5. Create Datasets ---
        train_ds, valid_ds = create_datasets(train_df, valid_df, char_to_num, max_label_length)
        
        # --- 6. Build, Compile, and Train Model ---
        model, history = build_and_train_model(train_ds, valid_ds, n_classes, char_to_num)
        
        # --- 7. Evaluate Model ---
        evaluate_model(model, valid_df, num_to_char)
        
        # --- 8. Save Final Model ---
        log.info(f"Saving final trained model to {MODEL_SAVE_PATH}...")
        model.save(MODEL_SAVE_PATH)
        
        # --- 9. Plot Training History ---
        plot_training_history(history)
        
        # --- 10. Save Final Results ---
        final_results = {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['loss']),
            'vocabulary_size': n_classes,
            'max_label_length': max_label_length
        }
        
        results_path = os.path.join(PLOTS_DIR, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        log.info(f"Training results saved to {results_path}")
        log.info(f"Final Results: {final_results}")
        
        log.info("=" * 60)
        log.info("Training Pipeline Complete!")
        log.info(f"Model saved to: {MODEL_SAVE_PATH}")
        log.info(f"Vocabulary saved to: {VOCAB_SAVE_PATH}")
        log.info(f"Training plots saved to: {PLOTS_DIR}")
        log.info("=" * 60)
        
    except Exception as e:
        log.error("Training pipeline failed with error:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()