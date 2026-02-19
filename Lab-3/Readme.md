# Lab-3 : NanoOwl Vision Transformer

## ⚠️ IMPORTANT SETUP INSTRUCTIONS

**Please DO NOT use Headless Mode** as it creates compatibility issues with display forwarding and GUI applications.

### Before Starting the Lab:

1. **Connect all peripherals to your Jetson Orin Nano:**
   - Power cable
   - DisplayPort (DP) cable
   - Ethernet cable
   - Keyboard & Mouse
   - USB Webcam

2. **Set Jetson to Maximum Power Mode:**
   - Click the **power icon** in the **top-right corner** of the desktop
   - Select **MAXN SUPER** power mode
   - This ensures maximum performance for model training and inference

**Power Mode Menu Reference:**

![Power Mode Setup](power_mode_setup.jpg)

> **Note:** These setup steps are crucial for proper operation of OpenCV GUI windows, webcam access, and optimal performance during training.

---

## Step 1: System Update

Open a terminal and run the following commands to update the system:
```bash
sudo apt update
sudo apt upgrade
```

When prompted, enter the machine password:
```
machinelearning<kit#>
```

> **Note:** The upgrade step may take a few minutes depending on the number of packages to update. Wait for it to complete fully before proceeding.
---

## Step 2: Pull the Docker Image

Run the following command to pull the NanoOWL Docker image:
```bash
sudo docker pull dustynv/nanoowl:r36.4.0
```

> **Note:** This is a large image (~6GB) and will take several minutes depending on your network speed. Wait for it to complete fully before proceeding.

---

## Step 3: Run the Docker Container

First, create the output directory on your host machine:
```bash
mkdir -p /home/$USER/nanoowl_outputs
```

Then run the Docker container:
```bash
sudo docker run -it --rm \
  --runtime nvidia \
  --device /dev/video0 \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -v /home/$USER/nanoowl_outputs:/outputs \
  --workdir /opt/nanoowl/examples/tree_demo \
  dustynv/nanoowl:r36.4.0 \
  /bin/bash
```

You will know you are inside the container when your terminal prompt changes to:
```
root@ubuntu:/opt/nanoowl/examples/tree_demo#
```

> **Note:** The `/outputs` folder is shared between the container and your host machine. Any files saved to `/outputs` inside the container will be accessible on your desktop at `/home/$USER/nanoowl_outputs`.

---

## Step 4: Install Required Module

Once inside the container, install the `aiohttp` module:
```bash
pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  aiohttp
```

> **Note:** This installs `aiohttp` which is required for the web server that streams the detection output to your browser. And everytime you exit the docker make sure to reinstall this module. It is a very small module so should be over within 1 minute.

---

## Step 5: Set Up Model Cache Directory

Run the following commands to set up the cache directory for the ViT model:
```bash
rm /root/.cache/clip
mkdir -p /root/.cache/clip
```

> **Note:** The first command removes an incorrectly created file that conflicts with the model cache, and the second creates a proper directory in its place. This is required for the model weights to download and store correctly.

---

## Step 6: Run the NanoOWL Tree Demo

Run the following command to start the ViT detection server:
```bash
python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
```

You will know the server is running successfully when you see a URL in the terminal like:
```
======== Running on http://0.0.0.0:7860 ========
```

Hold `Ctrl` and click the link to open it in your browser.

![NanoOWL Browser Demo](jetson_person_2x.gif)

NanoOWL is a project that optimizes **OWL-ViT** to run 🔥 real-time 🔥 on **NVIDIA Jetson Orin Platforms** with **NVIDIA TensorRT**. NanoOWL also introduces a new "tree detection" pipeline that combines OWL-ViT and CLIP to enable nested detection and classification of anything, at any level, simply by providing text.

Type whatever prompt you like to see what works! Here are some examples:
* Example: `[a face [a nose, an eye, a mouth]]`
* Example: `[a face (interested, yawning / bored)]`
* Example: `(indoors, outdoors)`

> **Note:** If the webcam feed is not displayed, reload the browser and it should appear.

---

## Lab-3 TODOs

In this lab you will move beyond simply running the model — you will **interrogate it**. Using NanoOWL and the attention visualization script, you will conduct a series of structured experiments designed to reveal how the model thinks, where it succeeds, and critically, where it fails.

The underlying goal is to develop intuition for how Vision Transformers differ from CNNs — not by reading about it, but by observing it directly through the model's behavior under controlled conditions. By the end of the lab you should be able to explain *why* OWL-ViT responds the way it does to different prompts, lighting conditions, and scene compositions, grounded in what you know about its architecture.

---

### Part 1 — Setup & Baseline

Before running any experiments, you need to establish a baseline. This gives you a reference point to compare against in all subsequent parts.

**Step 1:** Make sure the tree_demo server is running and open in your browser. Point the webcam at yourself and type the following prompt:
```
[a face]
```

Record the following in your lab notebook:
- Detection score displayed on the bounding box
- Which region of the image the box covers

**Step 2:** Now we will run the attention heatmap script to visualize what the model is focusing on. Copy the following code:
```python
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import cv2

CAMERA_DEVICE = 0

camera = cv2.VideoCapture(CAMERA_DEVICE)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")

print("Camera opened. Press ENTER to capture...")
input()

re, frame = camera.read()
if not re:
    raise RuntimeError("Failed to read frame")

camera.release()
print("Frame captured.")

cv2.imwrite("captured_frame.jpg", frame)
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
img_np = np.array(image)
img_w, img_h = image.size

print("Loading model...")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
model.eval()

prompts = [
    "a human face"
]

def run_prompt(prompt):
    inputs = processor(text=[[prompt]], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    logits = torch.sigmoid(outputs.logits[0, :, 0])
    boxes = outputs.pred_boxes[0]
    best_idx = logits.argmax().item()
    best_score = logits[best_idx].item()
    best_box = boxes[best_idx].detach().numpy()

    cx, cy, w, h = best_box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)

    similarity = torch.sigmoid(outputs.logits[0, :, 0]).detach().numpy()
    p_low, p_high = np.percentile(similarity, 10), np.percentile(similarity, 99)
    similarity = np.clip((similarity - p_low) / (p_high - p_low + 1e-6), 0, 1)
    num_patches = int(similarity.shape[0] ** 0.5)
    attn_map = similarity.reshape(num_patches, num_patches)

    attn_resized = cv2.resize(attn_map, (img_w, img_h))
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * img_np + 0.5 * heatmap_rgb).astype(np.uint8)

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, f"{best_score:.2f}", (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return attn_map, overlay, best_score

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, prompt in enumerate(prompts):
    print(f"Running prompt: '{prompt}'...")
    attn_map, overlay, score = run_prompt(prompt)

    axes[0].imshow(img_np)
    axes[0].set_title("Captured Image")
    axes[0].axis("off")

    axes[1].imshow(attn_map, cmap="hot")
    axes[1].set_title("Per-Patch Score")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Detection + Heatmap")
    axes[2].axis("off")

plt.tight_layout()
plt.savefig("attention_output.png", dpi=150)
print("Saved to attention_output.png")
```

In your Docker terminal, open the nano text editor:
```bash
nano attention_heatmap.py
```

Right-click inside the editor and select **Paste** to paste the code. Note that `Ctrl+C` and `Ctrl+V` do not work inside nano.

Once pasted, press `Ctrl+X` to exit. When prompted to save, press `Y`. You will be returned to the Docker terminal.

Verify the file was created by running:
```bash
ls
```

You should see `attention_heatmap.py` in the list. Then run the script:
```bash
python3 attention_heatmap.py
```

Point the webcam towards your face. Once you are satisfied with the angle, press `Enter` to capture the image. The model will then process it — you will know it is done when you see:
```
Saved to attention_output.png
```

Open a **new terminal** (do not close the existing one) and run:
```bash
sudo docker cp $(sudo docker ps -q):/opt/nanoowl/examples/tree_demo/attention_output.png ~/
```

This transfers the output image to your Jetson home directory where you can open and view the heatmap overlaid on your captured image.

**Step 3:** Fill in the baseline row of your results table:

| Prompt | Detection Score | Box Location | Heatmap Concentration |
|---|---|---|---|
| a human face | | | |

