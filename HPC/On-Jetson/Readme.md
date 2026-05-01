# FastVLM-0.5B on NVIDIA Jetson Orin Nano 8GB

This guide shows how to run Apple's **FastVLM-0.5B** vision-language model on an
NVIDIA Jetson Orin Nano 8GB using the `dustynv/l4t-pytorch:r36.4.0` Docker
container.

FastVLM-0.5B can describe images and answer image-based prompts. It is small
enough to run on the Jetson Orin Nano, but the setup needs a few Jetson-specific
fixes.

## What You Will Build

By the end, you will be able to run:

```bash
python3 predict.py \
  --model-path /workspace/fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
  --image-file /tmp/test.jpg \
  --prompt "Describe this image in detail."
```

Expected result: the model prints a natural-language description of the image.

## Hardware and Software

Tested configuration:

| Item | Version |
| --- | --- |
| Device | NVIDIA Jetson Orin Nano 8GB |
| Container | `dustynv/l4t-pytorch:r36.4.0` |
| Model | `apple/FastVLM-0.5B` |
| Repo | `https://github.com/apple/ml-fastvlm` |
| Verified | April 2026 |

FastVLM-0.5B details:

- About 759M parameters
- Uses about 1.5 GB VRAM in fp16
- Jetson Orin Nano 8GB has about 7.4 GB unified GPU memory
- Uses the `llava_qwen2` architecture with FastViTHD and Qwen2-0.5B

## Table of Contents

1. [Important JetPack Check](#1-important-jetpack-check)
2. [Free Memory on the Jetson](#2-free-memory-on-the-jetson)
3. [Launch the Docker Container](#3-launch-the-docker-container)
4. [Fix the Pip Index](#4-fix-the-pip-index)
5. [Clone FastVLM](#5-clone-fastvlm)
6. [Install Dependencies](#6-install-dependencies)
7. [Download the Model](#7-download-the-model)
8. [Patch the Code for Jetson CUDA](#8-patch-the-code-for-jetson-cuda)
9. [Run Inference](#9-run-inference)
10. [Performance Tips](#10-performance-tips)
11. [Troubleshooting](#11-troubleshooting)
12. [Working Package Versions](#12-working-package-versions)

## 1. Important JetPack Check

Do this before anything else.

JetPack `r36.4.7` has a known NVIDIA kernel issue that can break large CUDA
memory allocations. If your Jetson is on `r36.4.7`, FastVLM may fail with errors
like:

```text
NvMapMemAllocInternalTagged: 1075072515 error 12
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED
```

Check your JetPack version:

```bash
cat /etc/nv_tegra_release
```

Look for the `REVISION` value:

- `REVISION: 4.7`: downgrade to `r36.4.4` before continuing
- `REVISION: 4.4` or lower: continue to the next section

### If You Need to Downgrade from r36.4.7 to r36.4.4

Run this on the Jetson host, not inside Docker:

```bash
sudo apt-get install --allow-downgrades -y \
  nvidia-l4t-3d-core=36.4.4-20250616085344 \
  nvidia-l4t-apt-source=36.4.4-20250616085344 \
  nvidia-l4t-bootloader=36.4.4-20250616085344 \
  nvidia-l4t-camera=36.4.4-20250616085344 \
  nvidia-l4t-configs=36.4.4-20250616085344 \
  nvidia-l4t-core=36.4.4-20250616085344 \
  nvidia-l4t-cuda=36.4.4-20250616085344 \
  nvidia-l4t-cuda-utils=36.4.4-20250616085344 \
  nvidia-l4t-display-kernel=5.15.148-tegra-36.4.4-20250616085344 \
  nvidia-l4t-dla-compiler=36.4.4-20250616085344 \
  nvidia-l4t-firmware=36.4.4-20250616085344 \
  nvidia-l4t-gbm=36.4.4-20250616085344 \
  nvidia-l4t-graphics-demos=36.4.4-20250616085344 \
  nvidia-l4t-gstreamer=36.4.4-20250616085344 \
  nvidia-l4t-init=36.4.4-20250616085344 \
  nvidia-l4t-initrd=36.4.4-20250616085344 \
  nvidia-l4t-jetson-io=36.4.4-20250616085344 \
  nvidia-l4t-jetson-multimedia-api=36.4.4-20250616085344 \
  nvidia-l4t-jetsonpower-gui-tools=36.4.4-20250616085344 \
  nvidia-l4t-kernel=5.15.148-tegra-36.4.4-20250616085344 \
  nvidia-l4t-kernel-dtbs=5.15.148-tegra-36.4.4-20250616085344 \
  nvidia-l4t-kernel-headers=5.15.148-tegra-36.4.4-20250616085344 \
  nvidia-l4t-kernel-oot-headers=5.15.148-tegra-36.4.4-20250616085344 \
  nvidia-l4t-kernel-oot-modules=5.15.148-tegra-36.4.4-20250616085344 \
  nvidia-l4t-libwayland-client0=36.4.4-20250616085344 \
  nvidia-l4t-libwayland-cursor0=36.4.4-20250616085344 \
  nvidia-l4t-libwayland-egl1=36.4.4-20250616085344 \
  nvidia-l4t-libwayland-server0=36.4.4-20250616085344 \
  nvidia-l4t-multimedia=36.4.4-20250616085344 \
  nvidia-l4t-multimedia-utils=36.4.4-20250616085344 \
  nvidia-l4t-nvfancontrol=36.4.4-20250616085344 \
  nvidia-l4t-nvml=36.4.4-20250616085344 \
  nvidia-l4t-nvpmodel=36.4.4-20250616085344 \
  nvidia-l4t-nvpmodel-gui-tools=36.4.4-20250616085344 \
  nvidia-l4t-nvsci=36.4.4-20250616085344 \
  nvidia-l4t-oem-config=36.4.4-20250616085344 \
  nvidia-l4t-openwfd=36.4.4-20250616085344 \
  nvidia-l4t-optee=36.4.4-20250616085344 \
  nvidia-l4t-pva=36.4.4-20250616085344 \
  nvidia-l4t-tools=36.4.4-20250616085344 \
  nvidia-l4t-vulkan-sc=36.4.4-20250616085344 \
  nvidia-l4t-vulkan-sc-dev=36.4.4-20250616085344 \
  nvidia-l4t-vulkan-sc-samples=36.4.4-20250616085344 \
  nvidia-l4t-vulkan-sc-sdk=36.4.4-20250616085344 \
  nvidia-l4t-wayland=36.4.4-20250616085344 \
  nvidia-l4t-weston=36.4.4-20250616085344 \
  nvidia-l4t-x11=36.4.4-20250616085344 \
  nvidia-l4t-xusb-firmware=36.4.4-20250616085344
```

Then hold the packages so they do not auto-upgrade back to `r36.4.7`:

```bash
dpkg -l | grep nvidia-l4t | awk '{print $2}' | xargs sudo apt-mark hold
```

Reboot:

```bash
sudo reboot
```

After reboot, verify the version and test CUDA memory allocation:

```bash
cat /etc/nv_tegra_release
```

The output should show `REVISION: 4.4`.

```bash
python3 << 'EOF'
import torch
x = torch.zeros(400_000_000, dtype=torch.float16, device='cuda')
print(f"Allocated {x.nbytes/1024**3:.2f} GB - allocator OK")
del x
EOF
```

Expected output:

```text
Allocated 0.75 GB - allocator OK
```

Only continue after this test passes.

## 2. Free Memory on the Jetson

The Jetson Orin Nano uses unified memory, which means the CPU and GPU share the
same 8 GB memory pool. If background programs use too much RAM, the model may
run out of GPU memory.

Run these commands on the Jetson host before starting Docker:

```bash
free -h
tegrastats
```

If you are using a desktop environment, stop the display manager to free memory:

```bash
sudo systemctl stop gdm3
sudo systemctl stop lightdm
```

Set the Jetson to maximum performance:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Check memory again:

```bash
free -h
```

Target: at least 5 GB available before launching Docker.

## 3. Launch the Docker Container

Run this on the Jetson host:

```bash
sudo docker run -it --rm \
  --runtime nvidia \
  --network host \
  -v /home/$USER/fastvlm:/workspace/fastvlm \
  dustynv/l4t-pytorch:r36.4.0 \
  bash
```

Important Docker flags:

| Flag | Meaning |
| --- | --- |
| `--runtime nvidia` | Gives the container access to CUDA |
| `--network host` | Lets the container download files from the network |
| `-v /home/$USER/fastvlm:/workspace/fastvlm` | Saves code and model files outside the container |

Inside the container, verify CUDA:

```bash
python3 << 'EOF'
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')
EOF
```

Expected output:

```text
torch: 2.4.0
CUDA available: True
Device: Orin
VRAM: 7.4 GB
```

If CUDA is not available, exit the container and relaunch it with
`--runtime nvidia`.

## 4. Fix the Pip Index

The `dustynv` container may set pip to use a Jetson-specific package index:

```text
PIP_INDEX_URL=http://jetson.webredirect.org/jp6/cu126
```

This index can be unreachable or missing packages. Unset it before installing
Python packages:

```bash
unset PIP_INDEX_URL
unset PIP_TRUSTED_HOST
```

Verify:

```bash
pip config debug
```

The `env_var` section should no longer show `PIP_INDEX_URL`.

Make this permanent inside the container:

```bash
cat >> /root/.bashrc << 'EOF'

# Use PyPI instead of the Jetson-specific pip index.
unset PIP_INDEX_URL
unset PIP_TRUSTED_HOST
EOF

source /root/.bashrc
```

Important: do not upgrade the Jetson-provided `torch` package. The container's
`torch==2.4.0` is compiled for Jetson CUDA. Installing a normal PyPI build of
PyTorch can break CUDA.

## 5. Clone FastVLM

Inside the container:

```bash
cd /workspace/fastvlm
git clone https://github.com/apple/ml-fastvlm.git
cd ml-fastvlm
```

## 6. Install Dependencies

Do not run plain `pip install -e .`.

Some optional dependencies are not needed for inference and may fail on Jetson.
Instead, register the `llava` package without dependencies:

```bash
pip install -e . --no-deps
```

Then install only the packages needed for inference:

```bash
pip install \
  --index-url https://pypi.org/simple/ \
  transformers==4.48.3 \
  tokenizers==0.21.0 \
  sentencepiece==0.1.99 \
  accelerate==0.27.2 \
  safetensors \
  einops==0.6.1 \
  einops-exts==0.0.4 \
  timm==1.0.15 \
  shortuuid \
  pydantic \
  huggingface-hub==0.23.4 \
  pillow \
  markdown2
```

Why these versions matter:

- `accelerate==0.27.2`: newer versions can use a loading path that runs out of
  memory on Jetson unified memory
- `huggingface-hub==0.23.4`: newer versions may reject local absolute paths as
  invalid repo IDs

Verify the install:

```bash
python3 -c "import llava; print('llava OK')"
python3 -c "import transformers; print('transformers:', transformers.__version__)"
```

## 7. Download the Model

Inside the container, make sure you are in the fastvlm workspace directory (not inside ml-fastvlm):

```bash
cd /workspace/fastvlm
mkdir -p checkpoints
```

Download the Stage 3 instruction-tuned checkpoint (~1.5 GB):

```bash
wget -O fastvlm_0.5b_stage3.zip \
  https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_0.5b_stage3.zip
```

Extract it:

```bash
unzip fastvlm_0.5b_stage3.zip -d checkpoints/
```

Verify the checkpoint:

```bash
ls checkpoints/llava-fastvithd_0.5b_stage3/
```

You should see files such as:

```text
added_tokens.json
config.json
generation_config.json
merges.txt
model.safetensors
special_tokens_map.json
tokenizer_config.json
trainer_state.json
training_args.bin
vocab.json
```

## 8. Patch the Code for Jetson CUDA

The original FastVLM repo is set up for Apple Silicon. On Jetson, you need two
patches:

1. Replace `predict.py` — changes all Apple `mps` device references to NVIDIA `cuda`
2. Create `patch_builder.py` and run it — fixes model loading on Jetson unified memory

### Patch 1: Replace `predict.py`

Open the file with nano:

```bash
nano /workspace/fastvlm/ml-fastvlm/predict.py
```

Clear the existing content (`Ctrl+K` repeatedly, or `Ctrl+A` then `Ctrl+K`), then
paste the following content exactly, save with `Ctrl+O`, and exit with `Ctrl+X`:

```python
#
# FastVLM predict.py — Patched for Jetson Orin Nano (CUDA)
# Original: apple/ml-fastvlm (Apple Silicon / MPS)
# Changes:  All device="mps" references replaced with device="cuda"
#           Explicit torch.float16 passed to load_pretrained_model
#           image_tensor explicitly moved to CUDA device
#
# Usage:
#   python3 predict.py \
#     --model-path /workspace/fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
#     --image-file /path/to/image.jpg \
#     --prompt "Describe this image in detail."
#

import os
import argparse
import torch
from PIL import Image
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def predict(args):
    model_path = os.path.expanduser(args.model_path)

    # Temporarily rename generation_config.json so model uses args instead
    generation_config = None
    if os.path.exists(os.path.join(model_path, "generation_config.json")):
        generation_config = os.path.join(model_path, ".generation_config.json")
        os.rename(
            os.path.join(model_path, "generation_config.json"),
            generation_config,
        )

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    # PATCH: device="cuda" (original was device="mps")
    # PATCH: torch_dtype=torch.float16 to halve VRAM usage on Jetson
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        device="cuda",
        torch_dtype=torch.float16,
    )

    device = torch.device("cuda")

    # Build prompt with image token
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # PATCH: .to(device) instead of .to(torch.device("mps"))
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    # Load and preprocess image
    image = Image.open(args.image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            # PATCH: explicit .to(device) instead of relying on mps placement
            images=image_tensor.unsqueeze(0).half().to(device),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

    # Restore generation config
    if generation_config is not None:
        os.rename(
            generation_config,
            os.path.join(model_path, "generation_config.json"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Absolute path to the model checkpoint directory")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True,
                        help="Path to the input image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.",
                        help="Text prompt for the VLM")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    predict(args)
```

### Patch 2: Create and run `patch_builder.py`

Create the patcher script with nano:

```bash
nano /workspace/fastvlm/ml-fastvlm/patch_builder.py
```

Paste the following content, save with `Ctrl+O`, exit with `Ctrl+X`:

```python
"""
patch_builder.py — Patches llava/model/builder.py for Jetson Orin Nano (CUDA)

Run this ONCE after cloning the apple/ml-fastvlm repo:
    python3 patch_builder.py

What it fixes:
  1. LlavaQwen2ForCausalLM.from_pretrained()
       - Removes low_cpu_mem_usage=True  (triggers accelerate meta-device OOM)
       - Removes device_map from kwargs  (prevents accelerate from touching placement)
       - Forces torch_dtype=torch.float16
       - Adds model.cuda() after CPU load

  2. model.resize_token_embeddings()
       - Guards the call so it only runs if vocab size actually changed
       - Unconditional resize allocates on CUDA and crashes on stressed allocator
"""

import os
import sys

BUILDER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "llava", "model", "builder.py"
)

if not os.path.exists(BUILDER_PATH):
    print(f"ERROR: builder.py not found at {BUILDER_PATH}")
    print("Make sure you run this script from inside the ml-fastvlm directory.")
    sys.exit(1)

with open(BUILDER_PATH, "r") as f:
    content = f.read()

# ── Patch 1: LlavaQwen2ForCausalLM.from_pretrained ──────────────────────────

OLD_LOAD = """                model = LlavaQwen2ForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )"""

NEW_LOAD = """                # Patched for Jetson: plain CPU load then manual .cuda()
                # Removes low_cpu_mem_usage=True and device_map to bypass
                # accelerate's meta-device path which OOMs on unified memory.
                _load_kwargs = {k: v for k, v in kwargs.items()
                               if k not in ('device_map', 'torch_dtype')}
                model = LlavaQwen2ForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=False,
                    torch_dtype=torch.float16,
                    **_load_kwargs
                )
                model = model.cuda()"""

# ── Patch 2: resize_token_embeddings ────────────────────────────────────────

OLD_RESIZE = "    model.resize_token_embeddings(len(tokenizer))"

NEW_RESIZE = """    # Patched for Jetson: only resize if vocab size actually changed.
    # Unconditional resize_token_embeddings allocates on CUDA and can crash
    # the allocator in stressed container sessions (repeated failed runs).
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))"""

# ── Apply patches ────────────────────────────────────────────────────────────

patched = False

if OLD_LOAD in content:
    content = content.replace(OLD_LOAD, NEW_LOAD)
    print("Patch 1 applied: LlavaQwen2ForCausalLM.from_pretrained -> CPU-first load")
    patched = True
else:
    print("Patch 1 NOT applied: from_pretrained pattern not found.")
    print("The repo may have been updated. Check builder.py manually.")

if OLD_RESIZE in content:
    content = content.replace(OLD_RESIZE, NEW_RESIZE)
    print("Patch 2 applied: resize_token_embeddings -> conditional guard")
    patched = True
else:
    print("Patch 2 NOT applied: resize_token_embeddings pattern not found.")
    print("The repo may have been updated. Check builder.py manually.")

if patched:
    with open(BUILDER_PATH, "w") as f:
        f.write(content)
    print(f"\nSaved patched builder.py to: {BUILDER_PATH}")
    print("Verify with: sed -n '125,185p' llava/model/builder.py")
else:
    print("\nNo patches were applied. builder.py is unchanged.")
    print("If you already applied the patches manually, this is expected.")
```

Now run the patcher:

```bash
cd /workspace/fastvlm/ml-fastvlm
python3 patch_builder.py
```

Expected output:

```text
Patch 1 applied: LlavaQwen2ForCausalLM.from_pretrained -> CPU-first load
Patch 2 applied: resize_token_embeddings -> conditional guard

Saved patched builder.py to: .../llava/model/builder.py
Verify with: sed -n '125,185p' llava/model/builder.py
```

Verify the patch:

```bash
sed -n '125,145p' /workspace/fastvlm/ml-fastvlm/llava/model/builder.py
```

You should see:

- `low_cpu_mem_usage=False`
- `torch_dtype=torch.float16`
- `model = model.cuda()`

## 9. Run Inference

First, create or download a test image.

Option A: download an image:

```bash
python3 << 'EOF'
import urllib.request
urllib.request.urlretrieve(
    'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=640',
    '/tmp/test.jpg'
)
print('Downloaded /tmp/test.jpg')
EOF
```

Option B: create a synthetic test image without internet:

```bash
python3 << 'EOF'
from PIL import Image, ImageDraw

img = Image.new('RGB', (640, 480), color=(135, 206, 235))
draw = ImageDraw.Draw(img)
draw.rectangle([100, 150, 540, 400], fill=(34, 139, 34))
draw.ellipse([220, 50, 420, 200], fill=(255, 223, 0))
img.save('/tmp/test.jpg')
print('Created /tmp/test.jpg')
EOF
```

Run FastVLM. Make sure you are inside the ml-fastvlm directory first:

```bash
cd /workspace/fastvlm/ml-fastvlm

python3 predict.py \
  --model-path /workspace/fastvlm/checkpoints/llava-fastvithd_0.5b_stage3 \
  --image-file /tmp/test.jpg \
  --prompt "Describe this image in detail."
```

Important: use the full absolute path for `--model-path`.

The first run may take 20 to 30 seconds because the model has to load. Later
runs are faster if the model stays loaded.

## 10. Performance Tips

- Run max clocks before launching Docker:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

- Keep the model in fp16. Do not switch to fp32.
- For repeated inference, keep the model loaded instead of restarting
  `predict.py` for every image.
- Watch memory from another terminal on the Jetson host:

```bash
watch -n 1 tegrastats
```

- If inference runs out of memory, reduce the number of generated tokens in
  `predict.py` from `256` to `128`.

## 11. Troubleshooting

### Pip tries to install from `jetson.webredirect.org`

Fix:

```bash
unset PIP_INDEX_URL
unset PIP_TRUSTED_HOST
```

### `HFValidationError: Repo id must be in the form ...`

Fix:

```bash
pip install "huggingface-hub==0.23.4"
```

Also use the full absolute path for `--model-path`.

### `ModuleNotFoundError: No module named 'llava'`

Fix:

```bash
cd /workspace/fastvlm/ml-fastvlm
pip install -e . --no-deps
```

### CUDA is not available

If this returns `False`:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

You probably launched Docker without `--runtime nvidia`. Exit the container and
start it again with the Docker command from this README.

### `torch.OutOfMemoryError`

Try these fixes:

- Make sure the `builder.py` patch was applied
- Stop background processes on the Jetson host
- Keep `torch_dtype=torch.float16`
- Restart the container after failed runs
- Check available CUDA memory:

```bash
python3 << 'EOF'
import torch
torch.cuda.init()
print(torch.cuda.mem_get_info())
EOF
```

### `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED`

This can have two common causes:

1. You are on JetPack `r36.4.7`
2. The model loading path is using too much memory

Fixes:

- If you are on JetPack `r36.4.7`, downgrade to `r36.4.4`
- Apply the `builder.py` patch
- Restart the Docker container after repeated crashes
- Try:

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python3 predict.py ...
```

### `TypeError: got multiple values for keyword argument 'torch_dtype'`

The `builder.py` patch must remove `torch_dtype` from `kwargs` before passing
`**_load_kwargs`.

Check that your patch includes:

```python
_load_kwargs = {k: v for k, v in kwargs.items()
               if k not in ('device_map', 'torch_dtype')}
```

### `wavedrom`, `coremltools`, or `gradio` install fails

Use:

```bash
pip install -e . --no-deps
```

These packages are not needed for local inference.

## 12. Working Package Versions

Working configuration as of April 2026:

| Package | Version | Notes |
| --- | --- | --- |
| `torch` | `2.4.0` | Jetson-compiled. Do not upgrade. |
| `torchvision` | `0.19.0` | Jetson-compiled. Do not upgrade. |
| `transformers` | `4.48.3` | Required by this setup. |
| `tokenizers` | `0.21.0` | Required by this setup. |
| `accelerate` | `0.27.2` | Newer versions may break loading on Jetson. |
| `huggingface-hub` | `0.23.4` | Newer versions may reject local paths. |
| `safetensors` | Recent version | Any recent version should work. |
| `einops` | `0.6.1` | Required by this setup. |
| `einops-exts` | `0.0.4` | Required by this setup. |
| `timm` | `1.0.15` | Required by this setup. |
| `sentencepiece` | `0.1.99` | Required by this setup. |
| `pillow` | `10.4.0` | Usually already in the container. |
| `pydantic` | `2.x` | Latest 2.x works. |

Packages intentionally not installed:

- `coremltools`: Apple Silicon export only
- `gradio`: demo web UI only
- `wavedrom`: optional markdown diagram dependency
- `bitsandbytes`: optional quantization
- `peft`: LoRA fine-tuning
- `scikit-learn`: evaluation scripts
- `fastapi` and `uvicorn`: serving API

## Files Changed from the Original Repo

Two files are patched from the original `apple/ml-fastvlm` repo:

| File | Change |
| --- | --- |
| `predict.py` | Changes Apple Silicon `mps` device usage to NVIDIA `cuda` |
| `llava/model/builder.py` | Avoids Jetson memory issues during model loading |

These are the minimum changes needed for FastVLM-0.5B inference on Jetson Orin
Nano.

## References

- FastVLM GitHub repo: https://github.com/apple/ml-fastvlm
- Model family: `apple/FastVLM-0.5B`
- Container: `dustynv/l4t-pytorch:r36.4.0`
