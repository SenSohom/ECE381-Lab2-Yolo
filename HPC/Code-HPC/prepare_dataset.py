"""
Prepares the shot dataset for FastVLM fine-tuning.

Actual dataset layout:
    DATASET_ROOT/
        Shot-1/          ← maps to your first shot label
            1/           ← 15 frames (IMG_*.JPG)
            2/
            ...
            500/
        Shot-2/
        Shot-3/
        Shot-4/
        Shot-5/

Outputs to OUTPUT_DIR:
    mosaics/        ← one frame-grid image per datapoint
    train.json
    val.json
    test.json
"""

import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = "/project/th36/ss4887/pickleball-vlm/Sample"
OUTPUT_DIR   = "/project/th36/ss4887/pickleball-vlm/output"
# ─────────────────────────────────────────────────────────────────────────────


# ==============================================================================
#  SPORT CONFIGURATION — Edit this section for your sport group
# ==============================================================================

SPORT_NAME = "Pickleball"   # Change to your sport e.g. "Badminton", "Tennis"

# Map dataset folder names → shot-type labels.
# Keys must exactly match the folder names inside DATASET_ROOT.
# Values are the human-readable shot names used in training and evaluation.
SHOT_LABEL_MAP = {
    "Shot-1": "Serve",
    "Shot-2": "Third-Shot-Drop",
    "Shot-3": "Dink",
    "Shot-4": "Drive",
    "Shot-5": "Volley",
}

# Ground-truth text descriptions the model learns to generate.
# Write 1-2 sentences per shot describing:
#   - What the shot is
#   - Key motion cues visible in the frame sequence (stance, swing, contact point)
# Keys must match the values in SHOT_LABEL_MAP above.
SHOT_DESCRIPTIONS = {
    "Serve": (
        "This is a serve. The sequence shows the player initiating the point by "
        "dropping the ball and striking it underhand, sending it diagonally across "
        "the net into the opposite service box to start the rally."
    ),
    "Third-Shot-Drop": (
        "This is a third-shot drop. The sequence shows the player hitting a soft, "
        "arcing shot from the baseline that lands in the opponent's kitchen "
        "(non-volley zone), neutralising the net advantage and allowing the "
        "hitting team to advance forward."
    ),
    "Dink": (
        "This is a dink. The sequence shows the player executing a short, controlled "
        "shot near the kitchen line with a compact swing, the ball barely clearing "
        "the net and landing softly in the opponent's non-volley zone."
    ),
    "Drive": (
        "This is a drive. The sequence shows the player hitting a fast, flat, "
        "aggressive groundstroke with significant power and a full swing, keeping "
        "the ball low over the net to put pressure on the opponent."
    ),
    "Volley": (
        "This is a volley. The sequence shows the player striking the ball out of "
        "the air before it bounces, typically near the kitchen line, using a short "
        "punching motion with minimal backswing to redirect the ball."
    ),
}

# ==============================================================================


FRAME_SIZE  = (224, 224)
GRID_COLS   = 5
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
IMAGE_EXTS  = {".jpg", ".jpeg", ".png"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def grid_dims(n_frames: int) -> tuple[int, int]:
    cols = min(n_frames, GRID_COLS)
    rows = math.ceil(n_frames / cols)
    return rows, cols


def build_mosaic(frame_paths: list[str], out_path: str) -> tuple[int, int]:
    rows, cols = grid_dims(len(frame_paths))
    mosaic = Image.new("RGB", (cols * FRAME_SIZE[0], rows * FRAME_SIZE[1]))
    for idx, fp in enumerate(frame_paths):
        row, col = divmod(idx, cols)
        img = Image.open(fp).convert("RGB").resize(FRAME_SIZE, Image.LANCZOS)
        mosaic.paste(img, (col * FRAME_SIZE[0], row * FRAME_SIZE[1]))
    mosaic.save(out_path, quality=92)
    return rows, cols


def make_question(n_frames: int) -> str:
    rows, cols = grid_dims(n_frames)
    return (
        f"<image>\n"
        f"The image shows a {rows}×{cols} grid of {n_frames} consecutive frames "
        f"captured from a {SPORT_NAME} rally. "
        f"Frames run left-to-right, top-to-bottom in temporal order. "
        f"What type of {SPORT_NAME} shot is the player performing? "
        f"Identify the shot and briefly describe the key motion cues visible in the sequence."
    )


def collect_datapoints(dataset_root: str) -> list[dict]:
    datapoints = []
    for folder_name, shot_label in SHOT_LABEL_MAP.items():
        class_dir = Path(dataset_root) / folder_name
        if not class_dir.exists():
            print(f"[WARN] Missing class directory: {class_dir}")
            continue

        for seq_dir in sorted(class_dir.iterdir(), key=lambda p: p.name):
            if not seq_dir.is_dir():
                continue

            frames = sorted(
                f for f in seq_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith(".")
            )
            if not frames:
                print(f"[WARN] No frames found in {seq_dir}, skipping.")
                continue

            datapoints.append({
                "folder_name": folder_name,
                "shot_label":  shot_label,
                "seq_id":      seq_dir.name,
                "frame_paths": [str(f) for f in frames],
            })

    return datapoints


def make_annotation(dp: dict, mosaic_rel: str) -> dict:
    n = len(dp["frame_paths"])
    return {
        "id": f"{dp['folder_name']}_{dp['seq_id']}",
        "image": mosaic_rel,
        "conversations": [
            {"from": "human", "value": make_question(n)},
            {"from": "gpt",   "value": SHOT_DESCRIPTIONS[dp["shot_label"]]},
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mosaic_dir = Path(OUTPUT_DIR) / "mosaics"
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    datapoints = collect_datapoints(DATASET_ROOT)
    print(f"Sport     : {SPORT_NAME}")
    print(f"Collected : {len(datapoints)} datapoints across {len(SHOT_LABEL_MAP)} classes.")

    counts = Counter(dp["shot_label"] for dp in datapoints)
    for label, count in sorted(counts.items()):
        print(f"  {label:<25s}: {count}")

    random.seed(42)
    random.shuffle(datapoints)

    n       = len(datapoints)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits  = {
        "train": datapoints[:n_train],
        "val":   datapoints[n_train : n_train + n_val],
        "test":  datapoints[n_train + n_val :],
    }

    for split_name, split_dps in splits.items():
        annotations = []
        for dp in split_dps:
            fname       = f"{dp['folder_name']}_{dp['seq_id']}.jpg"
            mosaic_path = mosaic_dir / fname
            if not mosaic_path.exists():
                build_mosaic(dp["frame_paths"], str(mosaic_path))
            annotations.append(make_annotation(dp, str(Path("mosaics") / fname)))

        out_path = Path(OUTPUT_DIR) / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"  {split_name:5s}: {len(annotations):4d} samples  →  {out_path}")

    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
