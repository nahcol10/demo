# Nepali OCR Deployment# Nepali OCR Deployment



This project contains a two-part system:This project contains a two-part system:

1. A script to train the OCR model (`train.py`).1.  A script to train the OCR model (`train.py`).

2. A Flask web server to deploy and serve the trained model for inference (`app.py`).2.  A Flask web server to deploy and serve the trained model for inference (`app.py`).



## Features## 1. How to Train the Model



- Text detection using CRAFT (Character Region Awareness for Text detection)You only need to do this once. This step will download all training data, run the full training process, and create two essential files: `ocr_model.h5` and `vocabulary.json`.

- OCR recognition using CNN-RNN-CTC model

- REST API for PDF to text conversion**Prerequisites:**

- Docker support for easy deployment* You must have `git` and Python 3.10+ installed.

- Training pipeline with data augmentation* System libraries: `sudo apt-get update && sudo apt-get install -y libjpeg-turbo8-dev zlib1g-dev graphviz fonts-noto-core`



## Project Structure**Steps:**

1.  Create a project folder and save all the files (`model.py`, `train.py`, etc.) inside it.

```2.  Install the required Python packages:

.    ```bash

├── app.py                  # Flask API server    pip install -r requirements.txt

├── model.py               # Core OCR model and preprocessing functions    ```

├── train.py               # Training pipeline3.  Run the training script:

├── testing.py             # API testing script    ```bash

├── requirements.txt       # Python dependencies    python train.py

├── Dockerfile            # Docker configuration    ```

├── docker-compose.yml    # Docker Compose configuration4.  After this script finishes, you will have two new files in your directory:

├── ocr_model.h5          # Trained model (generated after training)    * `ocr_model.h5`: The trained Keras model.

├── vocabulary.json       # Character vocabulary (generated after training)    * `vocabulary.json`: The character mapping file.

└── CRAFT-pytorch/        # CRAFT text detection (cloned automatically)

```## 2. How to Deploy the Inference Server



## PrerequisitesOnce you have the `ocr_model.h5` and `vocabulary.json` files, you can run the server.



### System Requirements### Method A: Run Locally with Flask

- Python 3.10 or higher1.  Make sure you have installed the requirements:

- Git    ```bash

- 4GB+ RAM recommended    pip install -r requirements.txt

- GPU recommended for training (optional for inference)    ```

2.  Run the Flask app:

### System Libraries    ```bash

```bash    flask run --host=0.0.0.0 --port=8080

sudo apt-get update && sudo apt-get install -y \    ```

    libjpeg-turbo-progs \    The server is now running and listening on port 8080.

    zlib1g-dev \

    graphviz \### Method B: Run with Docker (Recommended for Deployment)

    fonts-noto-core \1.  Build the Docker image (this will copy your `ocr_model.h5` and `vocabulary.json` files into the image):

    libgl1-mesa-glx \    ```bash

    libglib2.0-0    docker build -t nepali-ocr .

```    ```

2.  Run the Docker container:

## 1. How to Train the Model    ```bash

    docker run -p 8080:8080 -d --name ocr-service nepali-ocr

You only need to do this once. This step will download all training data, run the full training process, and create two essential files: `ocr_model.h5` and `vocabulary.json`.    ```

    The server is now running in a container.

### Steps:

## 3. How to Use the API

1. **Clone or download this repository:**

   ```bashSend a `POST` request to the `/predict` endpoint with your PDF file.

   git clone <repository-url>

   cd demo**Example using cURL:**

   ``````bash

curl -X POST -F "file=@/path/to/your/test_file.pdf" [http://127.0.0.1:8080/predict](http://127.0.0.1:8080/predict)# demo

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training script:**
   ```bash
   python train.py
   ```

   This will:
   - Download training data from Google Drive
   - Preprocess and augment images
   - Train the OCR model
   - Save `ocr_model.h5` and `vocabulary.json`
   - Generate training plots in `training_plots/`

5. **Training outputs:**
   - `ocr_model.h5`: The trained Keras model
   - `vocabulary.json`: Character mapping file
   - `training_plots/`: Training visualizations
   - `model_checkpoints/`: Model checkpoints during training

## 2. How to Deploy the Inference Server

Once you have the `ocr_model.h5` and `vocabulary.json` files, you can run the server.

### Method A: Run Locally with Python

1. **Ensure requirements are installed:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app:**
   ```bash
   python app.py
   ```
   
   Or using Flask CLI:
   ```bash
   flask run --host=0.0.0.0 --port=8080
   ```

3. **For production, use Gunicorn:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8080 app:app
   ```

   The server is now running on `http://localhost:8080`

### Method B: Run with Docker (Recommended for Deployment)

1. **Build the Docker image:**
   ```bash
   docker build -t nepali-ocr .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 8080:8080 -d --name ocr-service nepali-ocr
   ```

3. **Or use Docker Compose:**
   ```bash
   docker-compose up -d
   ```

4. **Check logs:**
   ```bash
   docker logs ocr-service
   ```

5. **Stop the container:**
   ```bash
   docker-compose down
   # or
   docker stop ocr-service
   ```

## 3. How to Use the API

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Extract text from PDF

### Example: Using cURL

```bash
curl -X POST -F "file=@/path/to/your/document.pdf" http://localhost:8080/predict
```

### Example: Using Python (testing.py)

```bash
# Test with default file (test_file.pdf)
python testing.py

# Test with custom PDF file
python testing.py path/to/your/document.pdf

# Test with custom API URL
python testing.py path/to/your/document.pdf http://your-server:8080
```

### Example: Using Python Requests

```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8080/predict",
        files={"file": f}
    )
    
if response.status_code == 200:
    result = response.json()
    print(result["text"])
else:
    print("Error:", response.json())
```

### Response Format

**Success (200):**
```json
{
  "text": "Extracted text from PDF..."
}
```

**Error (400/500):**
```json
{
  "error": "Error description"
}
```

## 4. API Health Check

Check if the API is running and the model is loaded:

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "ocr_model.h5",
  "vocab_path": "vocabulary.json"
}
```

## 5. Development

### Project Components

1. **model.py** - Core functionality:
   - CRAFT text detection setup
   - Image preprocessing
   - OCR model architecture (CNN-RNN-CTC)
   - PDF to image conversion
   - Bounding box extraction and sorting

2. **train.py** - Training pipeline:
   - Data download and preparation
   - Data augmentation (rotation, noise)
   - Model training with callbacks
   - Evaluation and visualization

3. **app.py** - Flask API:
   - Model loading and inference
   - PDF processing endpoint
   - Health check endpoint

### Customization

To modify the model architecture, edit the `build_ocr_model()` function in `model.py`.

To adjust training parameters, modify constants in `train.py`:
- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`

## 6. Troubleshooting

### Model not found error
Ensure you've run `python train.py` first to generate the model files.

### CRAFT weights download fails
Manually download from [Google Drive](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view) and place at `CRAFT-pytorch/weights/craft_mlt_25k.pth`.

### Out of memory during training
Reduce `BATCH_SIZE` in `train.py` or use a machine with more RAM/GPU memory.

### API connection refused
Make sure the server is running: `python app.py`

## 7. License

This project uses the CRAFT model which is licensed under MIT License. Please check the CRAFT-pytorch repository for details.

## 8. Acknowledgments

- [CRAFT: Character Region Awareness for Text detection](https://github.com/clovaai/CRAFT-pytorch)
- TensorFlow and Keras teams
