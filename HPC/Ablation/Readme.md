# Ablation Study — FastVLM Shot Identification

An ablation study systematically tests how individual hyperparameter choices affect model performance. Instead of guessing the best settings, you train the same model multiple times — changing one thing at a time — and compare the results.

---

## Overview

You will run **5 configurations** on **2 model sizes** (FastVLM-0.5B and FastVLM-1.5B), for a total of **10 training runs**. Each run is submitted as a separate job and saves its adapter to its own folder.

| Config | Learning Rate | LoRA r | LoRA alpha | Epochs | What it tests |
|---|---|---|---|---|---|
| `baseline` | 2e-4 | 16 | 32 | 15 | Reference point |
| `low_lr` | 1e-4 | 16 | 32 | 15 | Effect of halving the learning rate |
| `high_rank` | 2e-4 | 32 | 64 | 15 | More LoRA capacity (more trainable params) |
| `low_rank` | 2e-4 | 8 | 16 | 15 | Less LoRA capacity (faster, may underfit) |
| `more_epochs` | 2e-4 | 16 | 32 | 30 | Longer training — does it help or overfit? |

---

## What to Modify in `finetune.py`

There are **3 parameters** to change per run, all at the top of `finetune.py`:

```python
# ── Training Settings ──────────────────────────────────
MODEL_ID = "apple/FastVLM-0.5B"    # line 1: model size
CKPT_DIR = "Root/fastvlm-lora-checkpoint"  # line 2: output folder
```

And inside `TrainingArguments` in `main()`:

```python
num_train_epochs = 15      # line 3: epochs
learning_rate    = 2e-4    # line 4: learning rate
```

And inside `apply_lora()`:

```python
r           = 16    # line 5: LoRA rank
lora_alpha  = 32    # line 6: LoRA alpha
```

---

## Step-by-Step for Each Run

### Run 1 — `baseline` (FastVLM-0.5B)

**In `finetune.py`:**

```python
MODEL_ID = "apple/FastVLM-0.5B"
CKPT_DIR = "/course/2026/spring/ece/381/th36/UCID/checkpoints/0.5B/baseline"

# in apply_lora():
r          = 16
lora_alpha = 32

# in TrainingArguments:
num_train_epochs = 15
learning_rate    = 2e-4
```

**Submit:**
```bash
sbatch job.sh
```

---

### Run 2 — `low_lr` (FastVLM-0.5B)

**In `finetune.py`:**

```python
MODEL_ID = "apple/FastVLM-0.5B"
CKPT_DIR = "/course/2026/spring/ece/381/th36/UCID/checkpoints/0.5B/low_lr"

# in apply_lora():
r          = 16
lora_alpha = 32

# in TrainingArguments:
num_train_epochs = 15
learning_rate    = 1e-4    # ← changed
```

**Submit:**
```bash
sbatch job.sh
```

---

### Run 3 — `high_rank` (FastVLM-0.5B)

**In `finetune.py`:**

```python
MODEL_ID = "apple/FastVLM-0.5B"
CKPT_DIR = "/course/2026/spring/ece/381/th36/UCID/checkpoints/0.5B/high_rank"

# in apply_lora():
r          = 32    # ← changed
lora_alpha = 64    # ← changed

# in TrainingArguments:
num_train_epochs = 15
learning_rate    = 2e-4
```

**Submit:**
```bash
sbatch job.sh
```

---

### Run 4 — `low_rank` (FastVLM-0.5B)

**In `finetune.py`:**

```python
MODEL_ID = "apple/FastVLM-0.5B"
CKPT_DIR = "/course/2026/spring/ece/381/th36/UCID/checkpoints/0.5B/low_rank"

# in apply_lora():
r          = 8     # ← changed
lora_alpha = 16    # ← changed

# in TrainingArguments:
num_train_epochs = 15
learning_rate    = 2e-4
```

**Submit:**
```bash
sbatch job.sh
```

---

### Run 5 — `more_epochs` (FastVLM-0.5B)

**In `finetune.py`:**

```python
MODEL_ID = "apple/FastVLM-0.5B"
CKPT_DIR = "/course/2026/spring/ece/381/th36/UCID/checkpoints/0.5B/more_epochs"

# in apply_lora():
r          = 16
lora_alpha = 32

# in TrainingArguments:
num_train_epochs = 30    # ← changed
learning_rate    = 2e-4
```

**Submit:**
```bash
sbatch job.sh
```

---

### Runs 6–10 — Repeat for FastVLM-1.5B

Repeat all 5 runs above with one change — set `MODEL_ID` to the larger model and point `CKPT_DIR` to a `1.5B` folder:

```python
MODEL_ID = "apple/FastVLM-1.5B"    # ← only change this
CKPT_DIR = "/course/2026/spring/ece/381/th36/UCID/checkpoints/1.5B/baseline"
#                                                                 ^^^
#                                               change 0.5B → 1.5B for each run
```

Everything else (LoRA settings, epochs, learning rate) stays identical to the corresponding 0.5B run.

---

## Output Structure

After all 10 runs complete, your checkpoint directory will look like this:

```
checkpoints/
├── 0.5B/
│   ├── baseline/
│   ├── low_lr/
│   ├── high_rank/
│   ├── low_rank/
│   └── more_epochs/
└── 1.5B/
    ├── baseline/
    ├── low_lr/
    ├── high_rank/
    ├── low_rank/
    └── more_epochs/
```

---

## Reading the Results

Each run produces a log file named `gpu_job.<job_id>.out`. The validation loss is printed at the end of each epoch:

```
{'eval_loss': 0.3821, 'epoch': 15}
```

Collect the **best (lowest) validation loss** from each run's log and fill in the table below:

| Config | 0.5B Val Loss | 1.5B Val Loss |
|---|---|---|
| baseline | | |
| low_lr | | |
| high_rank | | |
| low_rank | | |
| more_epochs | | |

**Lower validation loss = better generalisation.**

The winning configuration from each model size should be used for your final production training run and reported in your project submission.

---

## Important Notes

- Wait for each job to **finish** before submitting the next one, or submit all at once if the cluster has capacity — each run is fully independent.
- Each run downloads the base model weights only once; subsequent runs use the local HuggingFace cache.
- The 1.5B runs take roughly **1.5× longer** than the 0.5B runs per epoch.
- Do **not** change `DATASET_ROOT`, `OUTPUT_DIR`, or any sport/shot configuration between runs — only change the 3 parameters listed above.
