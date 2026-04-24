"""
Fine-tunes FastVLM-0.5B on the pickleball shot dataset using LoRA.
Run prepare_dataset.py first to generate mosaics and annotation JSON files.

Training regime
---------------
  - Base model : apple/FastVLM-0.5B  (fp16)
  - Adapter    : LoRA r=16 on all LLM attention + MLP projection layers
  - Epochs     : 15
  - Batch size : 4 per GPU × 4 gradient accumulation steps = effective batch 16
  - LR         : 2e-4 with cosine decay
"""

import json
import os
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchvision import transforms
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "Root"
CKPT_DIR = "Root/fastvlm-lora-checkpoint"

# Set False for Windows (bitsandbytes 4-bit unreliable on Windows).
# fp16 on RTX 4080 16GB is fine for FastVLM-0.5B.
# Set True on Wulver (Linux) to save VRAM.
USE_4BIT = False

MODEL_ID = "apple/FastVLM-0.5B"
HF_TOKEN = os.environ.get("HF_TOKEN")   # set in wulver_job.sh or via `huggingface-cli login`

# ── Image Processor ───────────────────────────────────────────────────────────
# FastVLM-0.5B ships no preprocessor_config.json.
# FastViTHD expects 1024×1024 inputs with ImageNet normalization.

FASTVLM_IMAGE_SIZE = 1024
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_image_transform = transforms.Compose([
    transforms.Resize((FASTVLM_IMAGE_SIZE, FASTVLM_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class FastVLMProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, images=None, return_tensors="pt",
                 padding=False, truncation=True, max_length=512):
        encoding = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        if images is not None:
            imgs = images if isinstance(images, list) else [images]
            encoding["images"] = torch.stack([_image_transform(img) for img in imgs])
        return encoding

    def save_pretrained(self, path: str):
        self.tokenizer.save_pretrained(path)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PickleballDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_json: str, data_dir: str, processor):
        with open(annotation_json) as f:
            self.annotations = json.load(f)
        self.data_dir  = Path(data_dir)
        self.processor = processor

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item  = self.annotations[idx]
        image = Image.open(self.data_dir / item["image"]).convert("RGB")

        human_msg = item["conversations"][0]["value"]
        gpt_msg   = item["conversations"][1]["value"]

        full_prompt = f"USER: {human_msg}\nASSISTANT: {gpt_msg}"

        encoding = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        labels = encoding["input_ids"].clone()
        assistant_ids = self.processor.tokenizer.encode(
            "ASSISTANT:", add_special_tokens=False
        )
        seq = labels.tolist()
        for i in range(len(seq) - len(assistant_ids), -1, -1):
            if seq[i : i + len(assistant_ids)] == assistant_ids:
                labels[: i + len(assistant_ids)] = -100
                break

        encoding["labels"] = labels
        return encoding


# ── Data Collator ─────────────────────────────────────────────────────────────

class DataCollator:
    def __init__(self, processor):
        self.pad_id = processor.tokenizer.pad_token_id or 0

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].shape[0] for f in features)
        B       = len(features)

        input_ids      = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len,               dtype=torch.long)
        labels         = torch.full((B, max_len), -100,        dtype=torch.long)

        for i, f in enumerate(features):
            L = f["input_ids"].shape[0]
            input_ids[i, :L]      = f["input_ids"]
            attention_mask[i, :L] = f["attention_mask"]
            labels[i, :L]         = f["labels"]

        batch = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        if "images" in features[0]:
            batch["images"] = torch.stack([f["images"] for f in features])

        return batch


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model_and_processor():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=HF_TOKEN
    )
    processor = FastVLMProcessor(tokenizer)

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN,
        )

    return model, processor


def apply_lora(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    model, processor = load_model_and_processor()
    model = apply_lora(model)

    data_dir = Path(DATA_DIR)
    train_ds = PickleballDataset(data_dir / "train.json", data_dir, processor)
    val_ds   = PickleballDataset(data_dir / "val.json",   data_dir, processor)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=20,
        dataloader_num_workers=1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(processor),
    )

    trainer.train()

    model.save_pretrained(CKPT_DIR)
    processor.save_pretrained(CKPT_DIR)
    print(f"Adapter + processor saved to {CKPT_DIR}")


if __name__ == "__main__":
    main()
