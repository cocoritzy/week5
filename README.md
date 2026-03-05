Urban Sound Classification & Whisper Fine‑Tuning
===============================================

I built this project as an end‑to‑end audio machine learning pipeline that:

- Trains a **2D CNN (`UrbanCNN`)** for urban sound classification from spectrograms.
- Experiments with a **CNN + Transformer encoder** architecture for sequence modeling.
- Fine‑tunes **OpenAI Whisper** for speech transcription and exposes it via a **Gradio web UI**.

The goal is to show how I design, train and serve modern audio models in PyTorch, from research notebooks to a small web demo.

## Project Highlights

- **Urban sound classifier**: `CNN/UrbanCNN` model for classifying spectrograms into multiple sound classes.
- **Hybrid CNN‑Transformer encoder**: `Encoder/classification` model built from a 1D CNN front‑end and Transformer‑style encoder blocks.
- **Whisper fine‑tuning & demo app**: A Gradio interface that loads my fine‑tuned Whisper weights and performs fast, English‑only transcription.
- **Modular code**: Separate folders for CNN, encoder experiments and Whisper tooling.

## Repository Structure

- `CNN/`
  - `model_CNN.py`: Definition of the `UrbanCNN` convolutional network.
  - `class_CNN.py`, `load_data.py`, `train_CNN.ipynb`, `test.ipynb`: Training, evaluation and data loading utilities.
  - `cnn_3couches.pt`: One of my trained CNN checkpoints.
- `Encoder/`
  - `model_encoder.py`: CNN1D + Transformer `classification` model.
  - `class_CNN.py`, `train_encoder.ipynb`, `test_encoder.ipynb`: Encoder‑based training and testing.
  - `encoder.pt`: Trained encoder checkpoint.
- `Whisper/`
  - `fine_tune.py`: Script I use to fine‑tune Whisper on my own dataset.
  - `gradio_app.py`, `gradio_app_compare.py`: Gradio apps to demo the fine‑tuned Whisper model.
  - `record.py`: Utility for recording audio samples.
- Root‑level notebooks and scripts:
  - `train.ipynb`, `process_data_onedata.ipynb`, `class_CNN.py`, `model_CNN.py`, `load_data.py`: Earlier CNN experiments and data processing.

## Getting Started

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

Some sub‑folders (such as `CNN/requirements.txt`) may define additional or pinned dependencies; install those as needed.

### 3. Prepare data

This project assumes you have audio data (e.g. urban sound clips) organized in a way compatible with the data loaders in `CNN/load_data.py` and the notebooks.

- Update any paths in the notebooks and scripts to point to your local dataset.
- Generate spectrograms or features as required by the CNN models.

## Running the UrbanCNN Classifier

The main CNN architecture is defined in `CNN/model_CNN.py` as `UrbanCNN`.

You can:

- Use the notebooks (`train_CNN.ipynb`, `test.ipynb`) to train and evaluate the network.
- Load the checkpoint `cnn_3couches.pt` as a pretrained starting point or reference.

## Running the Encoder‑Based Classifier

The encoder‑based classifier is defined in `Encoder/model_encoder.py` as `classification`, which combines:

- A `CNN1D` feature extractor.
- Transformer‑style `EncoderBlock` layers.
- A final classifier head.

Train and evaluate this model using `train_encoder.ipynb` and `test_encoder.ipynb`, adjusting hyperparameters as needed.

## Running the Whisper + Gradio Demo

The Gradio demo is defined in `Whisper/gradio_app.py`. It:

- Loads the **Whisper tiny** base model.
- Loads my **fine‑tuned weights** from `whisper_finetuned/model.pt`.
- Exposes a web interface where you can record or upload audio and see the transcription.

To run the app:

```bash
cd Whisper
python gradio_app.py
```

Make sure:

- You have a `whisper_finetuned/model.pt` checkpoint at the expected path.
- Whisper and Gradio are installed (via `requirements.txt`).

## Planned Next Steps

- Add evaluation scripts with standardized metrics (accuracy, F1, confusion matrices).
- Package the models as reusable Python modules with CLI entry points.
- Containerize the Gradio app (e.g. with Docker) for easier deployment.
- Add tests for data loading and model components.