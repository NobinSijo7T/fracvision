# FracVision

> **Multimodal X-ray Fracture Analysis and Recovery Prediction**
> *DINOv2 · Mamba · RT-DETRv2 · MedSAM*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.14-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Research%20Only-red)

</div>

---

> ⚠️ **RESEARCH PROTOTYPE — NOT FOR CLINICAL USE**
>
> All model outputs must be reviewed by a qualified clinician before any clinical application. This system has **NOT** been clinically validated and must **NOT** be used for autonomous diagnosis, treatment planning, or clinical decision-making of any kind.

---

## 🚀 HOW TO RUN THE PROJECT (QUICK START)

> 💡 **Want to get up and running in under 2 minutes?** Follow these quick commands.

### Option A: Run Locally (2 Terminals)

#### 1️⃣ Terminal 1 — Start Backend (FastAPI on Port 8000)
```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
```bash
# Linux / macOS
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
*Backend API & Interactive Swagger Docs will be live at: **[http://localhost:8000/api/docs](http://localhost:8000/api/docs)***

---

#### 2️⃣ Terminal 2 — Start Frontend (Next.js on Port 3000)
```bash
cd frontend
npm run dev
```
*Web Application will be live at: **[http://localhost:3000](http://localhost:3000)***

---

### Option B: Run with Docker (1 Single Command)

```bash
# CPU mode:
docker compose --profile cpu up

# GPU mode (with NVIDIA Container Toolkit):
docker compose --profile gpu up
```
*Frontend opens at: **[http://localhost:3000](http://localhost:3000)***  
*Backend opens at: **[http://localhost:8000/api/docs](http://localhost:8000/api/docs)***

---

## Table of Contents

1. [🚀 How to Run the Project (Quick Start)](#-how-to-run-the-project-quick-start)
2. [Project Overview](#1-project-overview)
3. [Architecture](#2-architecture)
4. [Datasets](#3-datasets)
5. [Prerequisites](#4-prerequisites)
6. [Installation — Step by Step](#5-installation--step-by-step)
   - [5.1 Python Setup](#51-python-setup)
   - [5.2 PyTorch (GPU/CPU)](#52-pytorch-gpucpu)
   - [5.3 Project Dependencies](#53-project-dependencies)
   - [5.4 Node.js & Frontend](#54-nodejs--frontend)
7. [Dataset Setup](#6-dataset-setup)
8. [Configuration](#7-configuration)
9. [Running the Data Pipeline](#8-running-the-data-pipeline)
10. [Training](#9-training)
11. [Evaluation](#10-evaluation)
12. [Running the Backend API](#11-running-the-backend-api)
13. [Running the Frontend](#12-running-the-frontend)
14. [Docker (Full Stack)](#13-docker-full-stack)
15. [Project Structure](#14-project-structure)
16. [API Reference](#15-api-reference)
17. [Limitations & Constraints](#16-limitations--constraints)
18. [Troubleshooting](#17-troubleshooting)

---

## 1. Project Overview

FracVision is a modular, reproducible research prototype that performs end-to-end analysis of bone X-ray images. It chains four state-of-the-art 2023–2024 foundation models in a unified pipeline:

| Capability | Model | Year |
|---|---|---|
| Rich feature extraction from X-ray images | **DINOv2** | 2023 |
| Longitudinal healing-pattern analysis across serial studies | **Mamba SSM** | 2023 |
| Fracture detection and bounding-box localization | **RT-DETRv2** | 2024 |
| Pixel-level fracture region segmentation | **MedSAM** | 2024 |

**What FracVision does:**
- Detects and localizes fractures in X-ray images with confidence scores
- Segments fracture regions at pixel level using prompted SAM
- Classifies fracture characteristics (body part, view, fracture presence)
- Compares serial X-rays from the same patient using image registration and embedding similarity
- Models healing-stage progression across patient visit sequences
- Serves predictions via a FastAPI REST backend and Next.js web UI

**What FracVision deliberately does NOT do:**
- Predict recovery time (no clinical outcome labels available in either dataset)
- Provide clinical diagnoses or treatment recommendations
- Make autonomous medical decisions of any kind

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FracVision Pipeline                       │
│                                                                   │
│  X-ray Image (JPEG / PNG / DICOM)                                │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────┐                                             │
│  │  Preprocessing  │  CLAHE normalization, DICOM support,        │
│  │  preprocess.py  │  aspect-preserving resize, QC checks        │
│  └────────┬────────┘                                             │
│           │                                                       │
│    ┌──────┼──────────────────────┐                               │
│    │      │                      │                               │
│    ▼      ▼                      ▼                               │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  RT-DETRv2   │  │     DINOv2       │  │    MedSAM        │  │
│  │  Detector    │  │   Extractor      │  │   Segmentor      │  │
│  │              │  │                  │  │                  │  │
│  │ BBoxes +     │  │ CLS token +      │  │ Prompted by      │  │
│  │ Confidence   │  │ patch tokens     │  │ RT-DETRv2 bbox   │  │
│  └──────┬───────┘  └───────┬──────────┘  └───────┬──────────┘  │
│         │                  │                      │              │
│         │                  ▼                      │              │
│         │        ┌──────────────────┐             │              │
│         │        │ FractureClassifier│            │              │
│         │        │ (DINOv2 head)    │             │              │
│         │        │ Multi-label:     │             │              │
│         │        │ body part, view, │             │              │
│         │        │ fracture present │             │              │
│         │        └──────────────────┘             │              │
│         │                                          │              │
│         └──────────────┬───────────────────────────┘             │
│                        │                                          │
│                        ▼                                          │
│              ┌──────────────────┐                                │
│              │  Prediction      │  Bounding boxes + confidence   │
│              │  Result          │  Segmentation mask + area      │
│              │                  │  Classification labels + probs  │
│              │  clinically_     │  QC status + warnings          │
│              │  validated=False │  Mandatory disclaimer          │
│              └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Longitudinal Analysis Path                     │
│                                                                   │
│  Patient Study 1 → Patient Study 2 → Patient Study N            │
│       │                   │                   │                  │
│       ▼                   ▼                   ▼                  │
│  [DINOv2 emb]        [DINOv2 emb]        [DINOv2 emb]          │
│  [det. features]     [det. features]     [det. features]        │
│  [seg. features]     [seg. features]     [seg. features]        │
│  [time delta]        [time delta]        [time delta]           │
│  [clinical]          [clinical]          [clinical]             │
│       │                   │                   │                  │
│       └───────────────────┴───────────────────┘                  │
│                           │                                       │
│                           ▼                                       │
│               ┌───────────────────────┐                          │
│               │   Mamba SSM (2023)    │  Selective state space   │
│               │   Longitudinal Model  │  model — efficient for   │
│               │                       │  variable-length series  │
│               │  Output per visit:    │                          │
│               │  healing_stage_prob   │  (0=fracture_visible,    │
│               │  [0.82, 0.18]         │   1=not visible)         │
│               └───────────────────────┘                          │
│                                                                   │
│  ⛔ Recovery-time supervised prediction: DISABLED                │
│     (No verified time-to-union labels in either dataset)         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       System Architecture                        │
│                                                                   │
│  ┌──────────────┐    REST API    ┌───────────────────────────┐  │
│  │  Next.js 16  │ ◄────────────► │  FastAPI Backend          │  │
│  │  Frontend    │                │  (uvicorn)                │  │
│  │  Port 3000   │                │  Port 8000                │  │
│  └──────────────┘                └──────────┬────────────────┘  │
│                                             │                    │
│                                    ┌────────┴────────┐           │
│                                    │  ML Pipeline    │           │
│                                    │  (loaded once   │           │
│                                    │   at startup)   │           │
│                                    └────────┬────────┘           │
│                                             │                    │
│                              ┌──────────────┼──────────────┐    │
│                              │              │              │    │
│                         ┌────┴──┐     ┌────┴──┐    ┌──────┴┐  │
│                         │DINOv2 │     │RTDETR │    │MedSAM │  │
│                         └───────┘     └───────┘    └───────┘  │
│                                             │                    │
│                                    ┌────────┴────────┐           │
│                                    │   PostgreSQL    │           │
│                                    │   (SQLAlchemy)  │           │
│                                    └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Datasets

FracVision is trained and evaluated on two public datasets. You must download them manually and place them in the correct folder structure (described in [Section 6](#6-dataset-setup)).

### Dataset 1 — FracAtlas

| | |
|---|---|
| **Download** | 🔗 [https://figshare.com/articles/dataset/The_dataset/22363012](https://figshare.com/articles/dataset/The_dataset/22363012) |
| **Size** | ~1 GB |
| **Images** | 4,085 X-rays (fractured + non-fractured) |
| **Body parts** | Hand, leg, hip, shoulder |
| **Annotation formats** | COCO JSON masks, YOLO TXT bounding boxes, Pascal VOC XML |
| **Labels used** | `fractured`, `hand`, `leg`, `hip`, `shoulder`, `frontal`, `lateral`, `oblique` |
| **Use in FracVision** | Fracture classification, segmentation, detection |
| **Recovery labels** | ❌ Not available — recovery prediction disabled for this dataset |

### Dataset 2 — GRAZPEDWRI-DX

| | |
|---|---|
| **Download** | 🔗 [https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193) |
| **Size** | ~4 GB (images in `folder_structure.zip`) |
| **Images** | 20,328 pediatric wrist X-rays |
| **Annotation format** | Pascal VOC XML |
| **Labels used** | `fracture`, `pronatorsign`, `periostealreaction`, `metalfragment`, `bonelesion`, `foreignbody`, `softtissue`, `text` |
| **Longitudinal data** | ✅ Multiple follow-up studies per patient (used for Mamba training) |
| **Use in FracVision** | Fracture detection, longitudinal healing-stage classification |
| **Recovery labels** | ❌ Not available — no verified time-to-union labels → recovery prediction disabled |

---

## 4. Prerequisites

You need the following installed on your system **before** setting up FracVision:

| Tool | Required Version | Purpose |
|---|---|---|
| **Python** | 3.10, 3.11, or 3.12 | ML model, training, backend |
| **pip** | ≥ 23.0 | Python package manager |
| **Git** | Any | Clone the project |
| **Node.js** | 18 LTS or 20 LTS | Next.js frontend |
| **npm** | ≥ 9 (comes with Node) | Frontend packages |
| **CUDA toolkit** | 11.8 or 12.1 (optional) | GPU acceleration |
| **Docker** | 24+ (optional) | Containerised deployment |

---

## 5. Installation — Step by Step

### 5.1 Python Setup

#### Windows

1. Download Python 3.11 from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Run the installer — **check "Add Python to PATH"** before clicking Install
3. Open PowerShell and verify:

```powershell
python --version
# Python 3.11.x

pip --version
# pip 24.x from ...
```

#### macOS

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Verify
python3.11 --version
pip3.11 --version
```

#### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Verify
python3.11 --version
pip3.11 --version
```

---

### Clone the Repository

```bash
git clone <your-repo-url> FracVision
cd FracVision
```

Or if you already have the project folder:

```bash
cd path/to/FracVision
```

---

### Create a Virtual Environment

It is strongly recommended to use a virtual environment to isolate project dependencies.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

---

### 5.2 PyTorch (GPU/CPU)

PyTorch must be installed **separately** because the correct version depends on your hardware.

Visit **[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)** and select your OS, package manager, and CUDA version.

**GPU (CUDA 12.1) — Recommended for training:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (slow for training, fine for inference testing):**
```bash
pip install torch torchvision torchaudio
```

Verify PyTorch and CUDA:
```python
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

### 5.3 Project Dependencies

With your virtual environment active and PyTorch installed:

```bash
pip install -r requirements.txt
```

This installs:
- `transformers` — RT-DETRv2 via Hugging Face
- `segment-anything` — MedSAM backbone
- `monai` — medical imaging utilities
- `pydicom` — DICOM file reading
- `opencv-python-headless` — image processing
- `albumentations` — augmentation
- `scikit-learn`, `scipy` — evaluation metrics
- `fastapi`, `uvicorn` — REST API backend
- `sqlalchemy`, `alembic` — database ORM
- `pydantic` — data validation
- `mlflow` — experiment tracking
- `matplotlib` — plots and reports
- and all other dependencies

> **Optional — Mamba CUDA-optimized kernel (significant speedup on GPU):**
> ```bash
> pip install mamba-ssm causal-conv1d
> ```
> This requires a CUDA-capable GPU and the CUDA toolkit installed. If unavailable, FracVision automatically falls back to the pure-PyTorch Mamba implementation.

> **Optional — SimpleITK (for image registration in serial comparisons):**
> ```bash
> pip install SimpleITK
> ```

Install the project itself as an editable package:
```bash
pip install -e .
```

---

### 5.4 Node.js & Frontend

#### Install Node.js

**Windows:**
1. Download Node.js 20 LTS from [https://nodejs.org/](https://nodejs.org/)
2. Run the installer with default settings
3. Verify in a new PowerShell window:

```powershell
node --version
# v20.x.x

npm --version
# 10.x.x
```

**macOS:**
```bash
brew install node@20
node --version
npm --version
```

**Linux (Ubuntu / Debian):**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version
npm --version
```

#### Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 6. Dataset Setup

### Folder Structure

All datasets go inside the `model/datasets/` directory. The exact folder layout is critical — the adapters look for files in specific locations.

```
model/
└── datasets/
    ├── FracAtlas/                          ← Extract FracAtlas here
    │   ├── dataset.csv                     ← Patient/image metadata
    │   ├── images/
    │   │   ├── Fractured/                  ← Fractured X-ray JPEGs
    │   │   │   ├── IMG0001.jpg
    │   │   │   └── ...
    │   │   └── Non_fractured/              ← Non-fractured X-ray JPEGs
    │   │       ├── IMG0002.jpg
    │   │       └── ...
    │   └── Annotations/
    │       ├── COCO JSON/
    │       │   └── COCO_fracture_masks.json  ← COCO segmentation masks
    │       ├── YOLO/                          ← YOLO TXT bounding boxes
    │       │   ├── IMG0001.txt
    │       │   └── ...
    │       └── PASCAL VOC/                   ← Pascal VOC XML annotations
    │           ├── IMG0001.xml
    │           └── ...
    │
    └── GRAZPEDWRI-DX/                      ← Extract GRAZPEDWRI-DX here
        ├── dataset.csv                     ← Study/patient metadata
        ├── folder_structure.zip            ← Images (auto-extracted on first run)
        └── folder_structure/               ← Created automatically from zip
            ├── images/                     ← PNG X-ray images
            │   ├── 00001.png
            │   └── ...
            └── pascalvoc/                  ← Pascal VOC XML annotations
                ├── 00001.xml
                └── ...
```

### Step-by-Step Dataset Download

#### FracAtlas

1. Open [https://figshare.com/articles/dataset/The_dataset/22363012](https://figshare.com/articles/dataset/The_dataset/22363012)
2. Click **Download all** (or download individual files)
3. Extract the downloaded archive
4. Copy the contents so your directory matches this structure:
   ```
   model/datasets/FracAtlas/
   ├── dataset.csv
   ├── images/Fractured/
   ├── images/Non_fractured/
   └── Annotations/COCO JSON/COCO_fracture_masks.json
   └── Annotations/YOLO/
   └── Annotations/PASCAL VOC/
   ```

#### GRAZPEDWRI-DX

1. Open [https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193)
2. Download `dataset.csv` and `folder_structure.zip`
3. Place both files in `model/datasets/GRAZPEDWRI-DX/`:
   ```
   model/datasets/GRAZPEDWRI-DX/
   ├── dataset.csv
   └── folder_structure.zip
   ```
4. **The zip is extracted automatically** on first run by the GRAZPEDWRI adapter. You do not need to extract it manually.

### Verify Dataset Placement

Run the inspection script to confirm everything is in the right place:

```bash
python model/data/inspect_datasets.py
```

Expected output:
```
INFO — FracAtlas: Found 4,085 records
INFO — GRAZPEDWRI-DX: Extracting folder_structure.zip ...
INFO — GRAZPEDWRI-DX: Found 20,328 records
Reports saved to: model/reports/
  dataset_inventory.json
  dataset_quality_report.json
  dataset_summary.csv
```

---

## 7. Configuration

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Open `.env` in your editor and set:

```env
# Dataset paths (can be relative or absolute)
FRACATLAS_ROOT=model/datasets/FracAtlas
GRAZPEDWRI_ROOT=model/datasets/GRAZPEDWRI-DX

# MedSAM checkpoint (download separately — see below)
MEDSAM_CHECKPOINT=model/checkpoints/medsam.pth

# Database (SQLite for development, PostgreSQL for production)
DATABASE_URL=sqlite:///./fracvision_dev.db
# DATABASE_URL=postgresql://fracvision:password@localhost:5432/fracvision

# Optional: MLflow experiment tracking
MLFLOW_TRACKING_URI=http://localhost:5000
```

All training hyperparameters, model settings, and inference thresholds are in:
```
model/configs/config.yaml
```

You generally do not need to edit this file to get started.

### MedSAM Checkpoint (Optional)

MedSAM requires a pre-trained checkpoint that must be downloaded separately:

1. Visit [https://github.com/bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)
2. Download `medsam_vit_b.pth` (~ 375 MB) from their release page
3. Place it at `model/checkpoints/medsam.pth`
4. Set `MEDSAM_CHECKPOINT=model/checkpoints/medsam.pth` in your `.env`

> Without this checkpoint, segmentation returns a clearly-labeled empty placeholder mask instead of crashing.

---

## 8. Running the Data Pipeline

Run each step in order. Each step creates files needed by the next.

### Step 1 — Inspect Datasets

Generates quality reports and verifies dataset integrity:

```bash
python model/data/inspect_datasets.py
```

Output files in `model/reports/`:
- `dataset_inventory.json` — full record counts and label distributions
- `dataset_quality_report.json` — missing files, corrupt images, label anomalies
- `dataset_summary.csv` — per-record summary table

### Step 2 — Create Patient-Level Splits

Creates train / val / test splits with **no patient overlap** across splits (critical for avoiding data leakage):

```bash
python model/data/create_splits.py
```

Output files in `model/data/splits/`:
- `train.csv` — ~70% of patients
- `val.csv` — ~15% of patients
- `test.csv` — ~15% of patients (held out for final evaluation)

Each CSV has columns: `image_id`, `patient_id`, `dataset`, `split`, `image_path`, `labels`

### Step 3 — Build Serial Study Pairs

Identifies patients with multiple follow-up studies (GRAZPEDWRI-DX) and builds consecutive-pair tables for longitudinal training:

```bash
python model/data/pairing/build_pairs.py
```

Output files:
- `model/reports/grazpedwri_sequences.json` — per-patient study sequences
- `model/data/pair_cache/grazpedwri_consecutive_pairs.csv` — consecutive study pairs

---

## 9. Training

All training scripts support `--dry-run` mode which runs 1 epoch on 5 samples — use this first to verify everything works before committing to a full training run.

### Phase 1 — DINOv2 Fracture Classifier

Trains a multi-label classification head on top of frozen DINOv2 features. Uses FracAtlas labels.

```bash
# Dry run first (fast, verifies setup)
python model/training/train_classifier.py --dry-run

# Full training
python model/training/train_classifier.py
```

Saves checkpoint to: `model/checkpoints/dinov2_classifier_best.pth`

### Phase 2 — RT-DETRv2 Fracture Detector

Fine-tunes the RT-DETRv2 detection head on FracAtlas YOLO annotations and GRAZPEDWRI-DX Pascal VOC annotations.

```bash
# Dry run first
python model/training/train_detector.py --dry-run

# Full training
python model/training/train_detector.py
```

Saves checkpoint to: `model/checkpoints/rtdetrv2_best.pt`

### Phase 3 — Mamba Longitudinal Model

Trains the Mamba SSM on GRAZPEDWRI-DX patient sequences using `fracture_visible` label progression as the healing-stage signal.

```bash
# Dry run first (requires at least some GRAZPEDWRI patients with 2+ studies)
python model/training/train_mamba.py --dry-run

# Full training
python model/training/train_mamba.py
```

Saves checkpoint to: `model/checkpoints/mamba_longitudinal_best.pth`

> **⛔ Recovery-time prediction training is DISABLED** and cannot be enabled via config. Neither dataset provides verified time-to-union labels. The `RecoveryPredictionModule` is a documented stub that clearly labels all its outputs as synthetic test values.

### Training All Phases Sequentially

```bash
python model/training/train_classifier.py
python model/training/train_detector.py
python model/training/train_mamba.py
```

### Monitoring with MLflow (Optional)

Start MLflow server before training:

```bash
# In a separate terminal
mlflow server --host 0.0.0.0 --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser to monitor training metrics, compare runs, and view artifacts.

---

## 10. Evaluation

Evaluate all trained models on the held-out test split:

```bash
# Quick smoke evaluation (5 samples, fast)
python model/evaluation/evaluate.py --smoke

# Full evaluation on test split
python model/evaluation/evaluate.py
```

Output files in `model/reports/`:
- `metrics.json` — machine-readable metrics (AUROC, AUPRC, F1, Dice, mAP)
- `evaluation_summary.md` — human-readable summary report
- `confusion_matrix.png` — fracture detection confusion matrix
- `roc_curve.png` — per-class ROC curves

---

## 11. Running the Backend API

### Development Mode

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag restarts the server on code changes.

On startup, the server:
1. Loads the inference pipeline (DINOv2 + RT-DETRv2 + MedSAM)
2. Initializes the database (creates tables if they don't exist)
3. Starts accepting requests

**Interactive API documentation:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

**OpenAPI schema:** [http://localhost:8000/api/openapi.json](http://localhost:8000/api/openapi.json)

### Production Mode

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "ok",
  "pipeline_ready": true,
  "model_versions": {
    "dinov2": "dinov2_vits14",
    "rtdetrv2": "rtdetrv2_r50vd",
    "medsam": "loaded"
  },
  "disclaimer": "This output is produced by a research prototype..."
}
```

---

## 12. Running the Frontend

Make sure the backend is running on port 8000 before starting the frontend.

### Development Mode

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
cd frontend
npm run build
npm start
```

### Frontend Pages

| Route | Description |
|---|---|
| `/` | Home page — overview and quick links |
| `/analyze` | Upload X-ray → fracture detection + classification |
| `/compare` | Upload prior + current X-ray → serial comparison |
| `/results` | Evaluation metrics dashboard |
| `/history` | Patient study history (requires backend + DB) |
| `/about` | Model details, dataset info, tech stack |

---

## 13. Docker (Full Stack)

Docker provides the easiest way to run the entire stack without manual Python or Node.js setup.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### CPU Mode (Development)

```bash
docker compose --profile cpu up
```

### GPU Mode (Recommended for Training)

```bash
docker compose --profile gpu up
```

### With MLflow Tracking

```bash
docker compose --profile gpu --profile mlflow up
```

### Environment Variables for Docker

Create a `.env` file in the project root (Docker Compose reads it automatically):

```env
FRACATLAS_ROOT=model/datasets/FracAtlas
GRAZPEDWRI_ROOT=model/datasets/GRAZPEDWRI-DX
MEDSAM_CHECKPOINT=model/checkpoints/medsam.pth
POSTGRES_PASSWORD=your_secure_password_here
DATABASE_URL=postgresql://fracvision:your_secure_password_here@db:5432/fracvision
```

### Accessing Services

| Service | URL |
|---|---|
| Frontend | [http://localhost:3000](http://localhost:3000) |
| Backend API | [http://localhost:8000](http://localhost:8000) |
| API Docs | [http://localhost:8000/api/docs](http://localhost:8000/api/docs) |
| MLflow (if enabled) | [http://localhost:5000](http://localhost:5000) |
| PostgreSQL | `localhost:5432` |

### Stop All Services

```bash
docker compose down
```

---

## 14. Project Structure

```
FracVision/
│
├── model/                              # All ML code
│   ├── configs/
│   │   └── config.yaml                # Master configuration
│   │
│   ├── datasets/                       # ← Place datasets here
│   │   ├── FracAtlas/
│   │   └── GRAZPEDWRI-DX/
│   │
│   ├── data/
│   │   ├── adapters/
│   │   │   ├── base.py                # Abstract adapter interface
│   │   │   ├── fracatlas.py           # FracAtlas dataset adapter
│   │   │   ├── grazpedwri.py          # GRAZPEDWRI adapter (auto zip-extract)
│   │   │   ├── coco_json.py           # COCO JSON mask parser
│   │   │   └── xml.py                 # Pascal VOC XML parser
│   │   ├── preprocessing/
│   │   │   ├── preprocess.py          # Image normalisation, DICOM, QC
│   │   │   └── augmentation.py        # Bbox-safe training augmentation
│   │   ├── pairing/
│   │   │   └── build_pairs.py         # Serial study pair builder
│   │   ├── splits/                    # Generated train/val/test CSVs
│   │   ├── inspect_datasets.py        # Quality report generator
│   │   └── create_splits.py           # Patient-level split creator
│   │
│   ├── models/
│   │   ├── dinov2/
│   │   │   ├── extractor.py           # DINOv2 CLS + patch token extraction
│   │   │   └── classifier.py          # Multi-label MLP + focal loss + calibration
│   │   ├── rtdetrv2/
│   │   │   └── detector.py            # RT-DETRv2 wrapper (HuggingFace)
│   │   ├── medsam/
│   │   │   └── segmentor.py           # MedSAM wrapper (graceful fallback)
│   │   └── mamba/
│   │       ├── longitudinal.py        # Mamba SSM (CUDA + PyTorch fallback)
│   │       └── recovery.py            # ⛔ Recovery stub (training disabled)
│   │
│   ├── training/
│   │   ├── train_classifier.py        # DINOv2 classifier training
│   │   ├── train_detector.py          # RT-DETRv2 detector training
│   │   └── train_mamba.py             # Mamba longitudinal training
│   │
│   ├── inference/
│   │   ├── pipeline.py                # Full inference pipeline orchestrator
│   │   └── compare.py                 # Serial X-ray comparison module
│   │
│   ├── evaluation/
│   │   └── evaluate.py                # Metrics, plots, evaluation reports
│   │
│   ├── checkpoints/                   # Saved model weights (gitignored)
│   ├── reports/                       # Generated evaluation reports
│   └── logs/                          # Training logs
│
├── backend/                           # FastAPI REST backend
│   ├── main.py                        # FastAPI app + lifespan model loading
│   ├── api/
│   │   └── routes.py                  # /health /models /predict /compare
│   ├── schemas/
│   │   └── schemas.py                 # Pydantic request/response models
│   ├── database/
│   │   └── models.py                  # SQLAlchemy ORM (patients, studies, predictions)
│   └── core/
│       └── pipeline_state.py          # Shared pipeline singleton
│
├── frontend/                          # Next.js 16 web application
│   ├── app/
│   │   ├── layout.tsx                 # Root layout + Navbar
│   │   ├── globals.css                # Global styles (Tailwind v4)
│   │   ├── page.tsx                   # Home page
│   │   ├── analyze/page.tsx           # X-ray upload + analysis
│   │   ├── compare/page.tsx           # Serial comparison
│   │   ├── results/page.tsx           # Evaluation dashboard
│   │   ├── history/page.tsx           # Patient history
│   │   └── about/page.tsx             # About + model details
│   ├── components/
│   │   ├── Navbar.tsx                 # Navigation bar
│   │   ├── XrayOverlay.tsx            # Canvas bbox overlay component
│   │   └── ResultsPanel.tsx           # Detection + classification results
│   ├── Dockerfile                     # Multi-stage production build
│   └── package.json
│
├── tests/
│   ├── conftest.py                    # Shared pytest fixtures
│   ├── unit/
│   │   ├── test_adapters.py           # Adapter + label validity tests
│   │   ├── test_preprocessing.py      # Image preprocessing tests
│   │   └── test_models.py             # Model shape + gradient + gate tests
│   └── integration/
│       └── test_pipeline.py           # End-to-end + split integrity + API tests
│
├── docker/
│   └── Dockerfile.backend             # PyTorch CUDA backend image
├── docker-compose.yml                 # GPU/CPU/MLflow profiles
│
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package metadata + pytest config
├── .env.example                       # Environment variable template
└── README.md                          # This file
```

---

## 15. API Reference

All endpoints live under the `/api` prefix.

### `GET /api/health`

Returns service health and model loading status.

```bash
curl http://localhost:8000/api/health
```

### `GET /api/models`

Lists all loaded model components and their metadata.

```bash
curl http://localhost:8000/api/models
```

### `POST /api/predict`

Upload a single X-ray image and receive full fracture analysis.

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@/path/to/xray.jpg"
```

**Response fields:**
- `fracture_detected` — boolean
- `fracture_confidence` — float [0, 1]
- `detections[]` — list of `{label, bbox_xyxy, confidence}`
- `classification` — `{labels[], probabilities{}, confidence, calibrated}`
- `segmentation` — `{mask_available, area_pixels, mask_score, warning}`
- `qc_passed`, `qc_warnings[]` — image quality flags
- `stage_errors{}` — per-stage error messages (if any stage fails)
- `clinically_validated` — always `false`
- `disclaimer` — mandatory clinical disclaimer string

Accepted file types: `.jpg`, `.jpeg`, `.png`, `.dcm`
Maximum file size: 50 MB

### `POST /api/compare`

Upload prior and current X-rays to compare them.

```bash
curl -X POST http://localhost:8000/api/compare \
  -F "prior=@/path/to/prior.jpg" \
  -F "current=@/path/to/current.jpg"
```

**Response fields:**
- `registration_applied` — whether SimpleITK registration succeeded
- `embedding_similarity` — DINOv2 cosine similarity [0, 1]
- `pixel_level_metrics` — `{mean_abs_diff, max_abs_diff, pct_pixels_changed_*}`
- `mask_comparison` — `{prior_area_pixels, current_area_pixels, mask_overlap_dice, mask_overlap_iou}`
- `metric_warning` — explicit non-clinical-validation warning
- `disclaimer` — mandatory clinical disclaimer

---

## 16. Limitations & Constraints

| Limitation | Detail |
|---|---|
| **Not clinically validated** | Results must never be used for clinical decisions without qualified clinician oversight |
| **Recovery prediction disabled** | No verified time-to-union labels in FracAtlas or GRAZPEDWRI-DX |
| **No invented labels** | Classifier only predicts labels confirmed to exist in dataset annotations (e.g., no `spiral`, `transverse`, `comminuted`) |
| **GRAZPEDWRI-DX wrist-only** | Detection model trained on pediatric wrist images; generalisation to other body parts requires additional data |
| **MedSAM checkpoint required** | Without the checkpoint, segmentation returns empty placeholder masks |
| **GPU strongly recommended** | DINOv2 + RT-DETRv2 training is extremely slow on CPU |
| **Patient-level splits** | Enforced — no patient appears in more than one split |

---

## 17. Troubleshooting

### `No module named 'torch'`

PyTorch is not installed or not in the current environment.

```bash
# Make sure your venv is active
.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate      # macOS / Linux

# Install PyTorch for your setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### `CUDA available: False` (unexpected)

Your PyTorch was installed without CUDA support.

```bash
# Uninstall and reinstall with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### `FileNotFoundError: GRAZPEDWRI-DX folder_structure.zip`

The GRAZPEDWRI-DX zip file has not been downloaded or is in the wrong location.

```
Expected: model/datasets/GRAZPEDWRI-DX/folder_structure.zip
```

Download it from [https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193).

### `No records found in FracAtlas`

The FracAtlas `dataset.csv` is not found or images are in the wrong subfolder.

```
Expected: model/datasets/FracAtlas/dataset.csv
Expected images: model/datasets/FracAtlas/images/Fractured/*.jpg
```

### `Port 8000 is already in use`

Another process is using port 8000.

```bash
# Find and kill the process (Windows)
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# macOS / Linux
lsof -i :8000
kill -9 <pid>
```

Or start the backend on a different port:

```bash
uvicorn backend.main:app --reload --port 8001
```

Then update the frontend `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8001
```

### Next.js build error: `Cannot apply unknown utility class`

This occurs when using Tailwind CSS v4. The `globals.css` uses plain CSS instead of `@apply`. Make sure you have not modified `globals.css` to add `@apply` directives.

### `MedSAM checkpoint not found — returning placeholder mask`

This is a warning, not an error. Segmentation is disabled until you provide the checkpoint. See [MedSAM Checkpoint](#medsam-checkpoint-optional).

### Tests fail with `No FracAtlas image available`

The dataset has not been downloaded or placed correctly. Tests that require real images are automatically skipped when datasets are absent. Run:

```bash
python model/data/inspect_datasets.py
```
to verify dataset placement.

---

## Licence

Research and educational use only. Not for commercial or clinical deployment without explicit authorisation and clinical validation.
