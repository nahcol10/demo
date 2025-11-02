# Nepali OCR Deployment

This project contains a two-part system:
1.  A script to train the OCR model (`train.py`).
2.  A Flask web server to deploy and serve the trained model for inference (`app.py`).

## 1. How to Train the Model

You only need to do this once. This step will download all training data, run the full training process, and create two essential files: `ocr_model.h5` and `vocabulary.json`.

**Prerequisites:**
* You must have `git` and Python 3.10+ installed.
* System libraries: `sudo apt-get update && sudo apt-get install -y libjpeg-turbo8-dev zlib1g-dev graphviz fonts-noto-core`

**Steps:**
1.  Create a project folder and save all the files (`model.py`, `train.py`, etc.) inside it.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the training script:
    ```bash
    python train.py
    ```
4.  After this script finishes, you will have two new files in your directory:
    * `ocr_model.h5`: The trained Keras model.
    * `vocabulary.json`: The character mapping file.

## 2. How to Deploy the Inference Server

Once you have the `ocr_model.h5` and `vocabulary.json` files, you can run the server.

### Method A: Run Locally with Flask
1.  Make sure you have installed the requirements:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the Flask app:
    ```bash
    flask run --host=0.0.0.0 --port=8080
    ```
    The server is now running and listening on port 8080.

### Method B: Run with Docker (Recommended for Deployment)
1.  Build the Docker image (this will copy your `ocr_model.h5` and `vocabulary.json` files into the image):
    ```bash
    docker build -t nepali-ocr .
    ```
2.  Run the Docker container:
    ```bash
    docker run -p 8080:8080 -d --name ocr-service nepali-ocr
    ```
    The server is now running in a container.

## 3. How to Use the API

Send a `POST` request to the `/predict` endpoint with your PDF file.

**Example using cURL:**
```bash
curl -X POST -F "file=@/path/to/your/test_file.pdf" [http://127.0.0.1:8080/predict](http://127.0.0.1:8080/predict)# demo
