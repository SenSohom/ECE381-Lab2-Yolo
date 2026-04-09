"""
Inference with the fine-tuned FastVLM shot classifier.

Usage:
    # From a folder of frames:
    python inference.py /path/to/sequence_folder/

    # From a pre-built mosaic image:
    python inference.py /path/to/mosaic.jpg
"""

import math
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT_DIR      = "/project/th36/ss4887/pickleball-vlm/checkpoints"
BASE_MODEL_ID = "apple/FastVLM-0.5B"
HF_TOKEN      = os.environ.get("HF_TOKEN")
FRAME_SIZE    = (224, 224)
GRID_COLS     = 5

# Must match SPORT_NAME in prepare_dataset.py
SPORT_NAME = "Pickleball"


def build_mosaic(frame_dir: str) -> tuple[Image.Image, int, int]:
    """Build a frame-grid mosaic from all images in frame_dir. Returns (image, rows, cols)."""
    frames = sorted(
        f for f in Path(frame_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and not f.name.startswith(".")
    )
    if not frames:
        raise ValueError(f"No image frames found in {frame_dir}")

    cols = min(len(frames), GRID_COLS)
    rows = math.ceil(len(frames) / cols)

    mosaic = Image.new("RGB", (cols * FRAME_SIZE[0], rows * FRAME_SIZE[1]))
    for idx, fp in enumerate(frames):
        r, c = divmod(idx, cols)
        img = Image.open(fp).convert("RGB").resize(FRAME_SIZE, Image.LANCZOS)
        mosaic.paste(img, (c * FRAME_SIZE[0], r * FRAME_SIZE[1]))
    return mosaic, rows, cols


FASTVLM_IMAGE_SIZE = 1024
_image_transform = transforms.Compose([
    transforms.Resize((FASTVLM_IMAGE_SIZE, FASTVLM_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FastVLMProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, images=None, return_tensors="pt", **kwargs):
        encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        if images is not None:
            imgs = images if isinstance(images, list) else [images]
            encoding["images"] = torch.stack([_image_transform(img) for img in imgs])
        return encoding


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, trust_remote_code=True, token=HF_TOKEN
    )
    processor = FastVLMProcessor(tokenizer)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(base, CKPT_DIR)
    model.eval()
    return model, processor


def make_prompt(rows: int, cols: int) -> str:
    n = rows * cols
    return (
        f"USER: <image>\n"
        f"The image shows a {rows}×{cols} grid of {n} consecutive frames captured "
        f"from a {SPORT_NAME} rally. Frames run left-to-right, top-to-bottom in temporal order. "
        f"What type of {SPORT_NAME} shot is the player performing?\n"
        f"ASSISTANT:"
    )


def predict(image: Image.Image, rows: int, cols: int, model, processor) -> str:
    inputs = processor(
        text=make_prompt(rows, cols),
        images=image,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    decoded = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "ASSISTANT:" in decoded:
        decoded = decoded.split("ASSISTANT:")[-1].strip()
    return decoded


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <frame_folder_or_mosaic_image>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if input_path.is_dir():
        image, rows, cols = build_mosaic(str(input_path))
    else:
        image = Image.open(input_path).convert("RGB")
        # Assume the pre-built mosaic used the default GRID_COLS layout
        cols = GRID_COLS
        rows = math.ceil((image.height // FRAME_SIZE[1]) * cols / cols)

    print("Loading model...")
    model, processor = load_model()

    print("Running inference...")
    result = predict(image, rows, cols, model, processor)
    print(f"\nPrediction:\n{result}")


if __name__ == "__main__":
    main()
