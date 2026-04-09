# Code-FineTune — Sport Shot Identification with FastVLM

Fine-tune Apple's **FastVLM-0.5B** (or 1.5B) on a sport shot dataset using LoRA, then run inference on new sequences. Designed to run on the **NJIT Wulver HPC cluster**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Sport & Shot Configuration](#sport--shot-configuration)
- [HuggingFace Token Setup](#huggingface-token-setup)
- [Python Files](#python-files)
- [Paths to Update](#paths-to-update)
- [How to Run](#how-to-run)

---

## Project Overview

Each group receives a sport with **5 shot types** and a dataset of **500 labeled sequences per shot** (15 frames each). This pipeline:

1. Converts the raw frame sequences into mosaic images (all 15 frames arranged in a grid)
2. Fine-tunes FastVLM on those mosaics using LoRA
3. Evaluates the model on held-out test sequences

---

## Dataset Structure

Your dataset on Wulver must follow this exact layout:

```
YourDataset/
├── Shot-1/                  ← folder name must match SHOT_LABEL_MAP keys
│   ├── 1/
│   │   ├── frame_01.jpg
│   │   ├── frame_02.jpg
│   │   └── ... (15 frames total)
│   ├── 2/
│   └── ... (500 sub-folders)
├── Shot-2/
├── Shot-3/
├── Shot-4/
└── Shot-5/
```

**Rules:**
- Top-level folders must be named `Shot-1` through `Shot-5`
- Each sequence sub-folder must contain exactly **15 image frames**
- Frames can be `.jpg`, `.jpeg`, or `.png`
- Do not include other files (e.g. `.DS_Store`) inside sequence folders

---

## Sport & Shot Configuration

> **This is the only section you need to edit for your sport group.**

Open `prepare_dataset.py` and find the block labelled:

```
# ==============================================================================
#  SPORT CONFIGURATION — Edit this section for your sport group
# ==============================================================================
```

### 1. Set your sport name

```python
SPORT_NAME = "Pickleball"   # Change to e.g. "Badminton", "Tennis"
```

### 2. Map folder names to shot labels

```python
SHOT_LABEL_MAP = {
    "Shot-1": "Serve",           # ← replace with your shot names
    "Shot-2": "Third-Shot-Drop",
    "Shot-3": "Dink",
    "Shot-4": "Drive",
    "Shot-5": "Volley",
}
```

The keys (`Shot-1` … `Shot-5`) must exactly match the folder names in your dataset. The values are the human-readable names your model will learn to predict.

### 3. Write shot descriptions

```python
SHOT_DESCRIPTIONS = {
    "Serve": "This is a serve. The player ...",
    ...
}
```

Write **1–2 sentences** per shot that describe:
- What the shot is
- Key motion cues visible across the 15-frame sequence (stance, backswing, contact point, follow-through)

These descriptions are the ground-truth text the model trains to generate, so make them accurate and consistent.

### 4. Also update `inference.py`

Set the same sport name at the top of `inference.py`:

```python
SPORT_NAME = "Pickleball"   # must match prepare_dataset.py
```

---

## HuggingFace Token Setup

The model (`apple/FastVLM-0.5B`) is downloaded from HuggingFace during training and inference. You need a token with **Write** permissions.

### Create a token

1. Go to **https://huggingface.co** and sign in (create a free account if needed)
2. Click your profile picture → **Settings**
3. In the left sidebar click **Access Tokens**
4. Click **New token**
5. Give it a name (e.g. `wulver-fastvlm`)
6. Set the role to **Write** (required for saving models back to HF if needed; Read is sufficient for download only)
7. Click **Generate token** and copy it — you will not be able to see it again

### Add the token to the job script

Open `job.sh` and paste your token here:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## Python Files

### `prepare_dataset.py`

Reads the raw frame sequences from `DATASET_ROOT`, builds a mosaic image for each sequence (all frames arranged in a grid), splits the data into train / val / test sets (80 / 10 / 10), and writes annotation JSON files used by the fine-tuning script.

**Output:**
```
OUTPUT_DIR/
├── mosaics/          ← one .jpg mosaic per sequence
├── train.json
├── val.json
└── test.json
```

Run this **once** before fine-tuning. Re-run only if the dataset changes.

---

### `finetune.py`

Loads FastVLM-0.5B, wraps it with LoRA adapters, and trains on the mosaic dataset produced by `prepare_dataset.py`. Uses the HuggingFace `Trainer` with fp16 mixed precision.

Key training settings (edit in `finetune.py`):

| Parameter | Default | Description |
|---|---|---|
| `MODEL_ID` | `apple/FastVLM-0.5B` | Switch to `apple/FastVLM-1.5B` for larger model |
| `num_train_epochs` | `15` | Training epochs |
| `learning_rate` | `2e-4` | AdamW learning rate |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA scaling factor |

Only the **LoRA adapter** is saved (~30 MB), not the full model weights.

---

### `inference.py`

Loads the fine-tuned LoRA adapter on top of the base FastVLM model and runs prediction on a new sequence. Accepts either a folder of frames (builds the mosaic automatically) or a pre-built mosaic image.

```bash
# From a frame folder
python inference.py /path/to/sequence_folder/

# From a pre-built mosaic
python inference.py /path/to/mosaic.jpg
```

---

## Paths to Update

Before running on Wulver, update the following paths in each file:

| File | Variable | What to set |
|---|---|---|
| `prepare_dataset.py` | `DATASET_ROOT` | Path to your dataset folder on Wulver |
| `prepare_dataset.py` | `OUTPUT_DIR` | Where mosaics and JSON splits will be written |
| `finetune.py` | `DATA_DIR` | Same as `OUTPUT_DIR` above |
| `finetune.py` | `CKPT_DIR` | Where the trained LoRA adapter will be saved |
| `inference.py` | `CKPT_DIR` | Same as `CKPT_DIR` above |
| `job.sh` | `HF_TOKEN` | Your HuggingFace token |

---

## How to Run

### Step 1 — Navigate to your course directory

After logging in to Wulver, go to your personal course directory (replace `UCID` with your actual UCID):

```bash
cd /course/2026/spring/ece/381/th36/UCID
```

---

### Step 2 — Load modules

```bash
module load Miniforge3/24.11.3-0
module load CUDA/12.6.0
```

---

### Step 3 — Create and activate the conda environment *(once only)*

Do this only the first time. Skip to Step 4 on subsequent runs.

```bash
conda create -n fine-tune python=3.10 -y
conda activate fine-tune
pip install -r requirements.txt
```

For all future sessions, just activate the existing environment:

```bash
conda activate fine-tune
```

---

### Step 4 — Submit the job

```bash
sbatch job.sh
```

---

### Step 5 — Monitor progress

```bash
squeue -u $USER                          # check job status
tail -f gpu_job.<job_id>.out             # live training log
```

---

### Step 6 — Run inference after training

```bash
python inference.py /path/to/test_sequence/
```
