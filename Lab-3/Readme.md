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

> **Note:** If the webcam feed is not displayed, reload the browser and it should appear.
