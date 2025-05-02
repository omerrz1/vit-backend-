
# Skin Cancer Scan API

This project provides a FastAPI-based server for scanning skin images to detect potential skin cancer types using a Vision Transformer (ViT) model. It includes explainability features like Grad-CAM and attention maps, stores scan results in a SQLite database, and allows for updating actual results for scanned images.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Database](#database)
- [Explainability](#explainability)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload skin images for quick scanning and classification.
- Supports classification of five skin conditions:
  - Actinic keratosis
  - Benign keratosis
  - Dermatofibroma
  - Melanocytic nevus
  - Vascular lesion
- Generates explainability visualizations (Grad-CAM and attention maps) for model predictions.
- Stores scan results and images in a SQLite database.
- Allows updating actual results for scans to facilitate result verification.
- Serves explainability images via a static file mount.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt` (see Setup below)
- CUDA-enabled GPU (optional, for faster model inference)

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/omerrz1/vit-backend-/
    cd vit-backend-
    ```

2. Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:


    install packages :

    ```bash
    pip install fastapi uvicorn sqlmodel pillow torch torchvision transformers captum matplotlib numpy scikit-learn datasets accelerate
    ```

4. Download the pre-trained model:

    Ensure the `trainedModel/` directory contains the pre-trained ViT model. If not, use `finetune.ipynb` to fine-tune it (see [Training](#training)).

5. Set up the database:

    The SQLite database (`database.db`) is automatically created when the server starts, with tables defined in `models.py`.

## Running the Server

Start the FastAPI server:

```bash
uvicorn server:app --reload
````

* The server runs at `http://127.0.0.1:8000`
* Use the `--reload` flag for automatic code reloads during development.

### Access the API

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to use the interactive Swagger UI.

## API Endpoints

### `POST /quick-scan/`

* Upload an image file to perform a skin cancer scan.
* Returns scan ID, classification results, and a path to the explainability image.

**Example:**

```bash
curl -X POST -F "file=@image.jpg" http://127.0.0.1:8000/quick-scan/
```

---

### `GET /scans/`

* Retrieve a list of all scans stored in the database.
* Returns scan IDs, results, and actual results (if available).

---

### `PUT /update-result/{scan_id}/`

* Update the actual result for a specific scan by ID.

**Example:**

```bash
curl -X PUT -F "actual_result=Benign keratosis" http://127.0.0.1:8000/update-result/1/
```

---

### Static Files `/explainability/`

* Access explainability images (e.g., Grad-CAM and attention maps):

```
http://127.0.0.1:8000/explainability/<filename>
```

## Project Structure

```
├── server.py              # FastAPI server implementation
├── scan.py                # Image scanning and explainability logic
├── db.py                  # Database setup and configuration
├── models.py              # SQLModel for database schema
├── finetune.ipynb         # Jupyter notebook for model training
├── trainedModel/          # Directory for pre-trained ViT model
├── explainability/        # Directory for explainability image outputs
├── database.db            # SQLite database (created on startup)
└── README.md              # This file
```

## Model Details

* **Model**: Vision Transformer (ViT) fine-tuned on the `ThankGod/melanoma` dataset
* **Classes**:

  * Actinic keratosis
  * Benign keratosis
  * Dermatofibroma
  * Melanocytic nevus
  * Vascular lesion
* **Explainability**: Grad-CAM and attention maps
* **Pre-trained Base**: `WinKawaks/vit-tiny-patch16-224`

## Database

* **Backend**: SQLite (`database.db`)
* **Schema**: Defined in `models.py` using `SkinCancerScan` model

Fields:

* `id`: Unique scan ID
* `image`: Raw image bytes
* `scan_result`: JSON string of model prediction
* `actual_result`: Optional updated result

## Explainability

* **Grad-CAM**: Highlights influential regions in image
* **Attention Maps**: Shows attention focus of the ViT model
* **Output**: Saved as `.png` in `explainability/`, accessible via `/explainability/` endpoint

## Training

Use `finetune.ipynb` for full fine-tuning workflow:

* Load `ThankGod/melanoma` dataset
* Preprocess with `AutoImageProcessor`
* Train with HuggingFace `Trainer`
* Evaluate with accuracy, precision, recall, F1
* Generate:

  * Confusion matrix
  * Attention maps
  * Grad-CAM images

**To run:**

```bash
jupyter notebook finetune.ipynb
```

Make sure all required packages are installed (listed in the first cell of the notebook)
