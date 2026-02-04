# Lab-2 : YOLO Manual

## Part 1: Docker Setup

### Pre-Step: Open Terminal
**Open a terminal on your Jetson Orin Nano.** All subsequent commands in this lab will be executed in this terminal.

---

### Pre-Step: Create Lab2-Workspace Folder
**Create a new working directory for this lab.**

> **Note:** Your Jetson kit has a number written on it (e.g., 1, 2, 3, etc.). Replace `<kit#>` with your kit number in the command below.

```bash
mkdir /home/ece381-<kit#>/Documents/Lab2-Workspace
```

**Example:** If your kit number is 15, the command would be:
```bash
mkdir /home/ece381-15/Documents/Lab2-Workspace
```

---

### Step 1: Enable X11 Display Forwarding
```bash
xhost +local:docker
```

### Step 2: Launch Docker Container
```bash
sudo docker run -it --ipc=host --runtime=nvidia --device=/dev/video0 \
  -v /home/ece381-<kit#>/Documents/Lab2-Workspace:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ultralytics/ultralytics:latest-jetson-jetpack6 bash
```

> **Note:** Replace `<kit#>` with your kit number (same as in the previous step).

### Step 3: Install Display Libraries
```bash
apt-get update && apt-get install -y libgtk2.0-dev libsm6 libxext6
```

### Step 4: Install Build Dependencies
```bash
apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
```

### Step 5: Clone and Build OpenCV with GTK Support
```bash
cd /tmp
git clone --depth 1 https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_GTK=ON ..
make -j4
make install
ldconfig
```

---

## Part 2: Running Object Detection

### Step 1: Navigate to Workspace
```bash
cd /workspace
```

---

### Step 2: Convert Model to TensorRT Engine Format
**Convert the YOLOv11n PyTorch model (.pt) to TensorRT engine format (.engine) for optimized inference on Jetson.**

Create a file named `ModelConversion.py`:
```python
from ultralytics import YOLO

# Load a YOLOv11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to TensorRT engine format
model.export(format="engine")  # creates 'yolo11n.engine'

# Load and verify the exported TensorRT model
trt_model = YOLO("yolo11n.engine")

# Run inference to verify
results = trt_model("https://ultralytics.com/images/bus.jpg")
```

Then run the conversion:
```bash
python ModelConversion.py
```

This will:
- Convert the PyTorch model to TensorRT engine format
- Optimize it for your Jetson's GPU architecture
- Provide faster inference times (22-25ms per frame)

---

### Step 3: Run Object Detection Script
```bash
python ObjectDetect.py
```

This will:
- Load the YOLOv11n TensorRT model
- Capture video from your webcam
- Display real-time object detection with bounding boxes
- Press 'q' to exit

---

## Part 3: Object Detection Task

### Task 3.1: Detect and Screenshot Objects
**Objective:** Identify 10 different objects around your desk using the pretrained YOLOv11n model.

**Instructions:**
1. Run `python ObjectDetect.py` (from Part 2)
2. Place 10 different objects on/around your desk
3. Capture 10 screenshots showing successful detections with confidence scores
4. Save screenshots with filenames: `detection_01.png`, `detection_02.png`, ..., `detection_10.png`
5. Include these in your lab report

---

### Task 3.2: Build Custom Datasets
**Objective:** Create three datasets with increasing sizes for training custom object detection models.

**Classes:** 
- Class 1: Oscilloscope
- Class 2: Jetson (Jetson Orin Nano board)

**Dataset Requirements:**
- **Dataset 1:** 10 images of Oscilloscope + 10 images of Jetson (20 total)
- **Dataset 2:** 25 images of Oscilloscope + 25 images of Jetson (50 total)
- **Dataset 3:** 50 images of Oscilloscope + 50 images of Jetson (100 total)

**Annotation Tool:** Use Roboflow for annotation and automatic YOLO format conversion
- Visit: https://roboflow.com
- Create account and project
- Upload images and draw bounding boxes
- Export in YOLOv11 format

**Refer to the video "Annotate-using-Roboflow.mp4"** to learn how to use the tool and download the custom dataset.

**After Exporting from Roboflow:**

The exported dataset will be downloaded as a **ZIP file** to your `Downloads` folder.

**Steps to extract and organize:**

1. **Open the Downloads folder** using the file manager (GUI)

2. **Locate the downloaded ZIP file**

3. **Right-click on the ZIP file and select "Extract Here"**

4. **Move the extracted folder to Lab2-Workspace:**
   - Open the file manager and navigate to `Documents/Lab2-Workspace`
   - Cut/paste the extracted folder into Lab2-Workspace
   - Rename the folder based on dataset size:
     - First dataset (10 images per class) → Rename to `Dataset-10`
     - Second dataset (25 images per class) → Rename to `Dataset-25`
     - Third dataset (50 images per class) → Rename to `Dataset-50`

> **Note:** After extracting, you should have three folders (Dataset-10, Dataset-25, Dataset-50) in your Lab2-Workspace.

**Expected Directory Structure after extraction:**
```
Lab2-Workspace/
├── Dataset-10/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── Dataset-25/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
└── Dataset-50/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

---

### Task 3.3: Train Models on Three Datasets
**Objective:** Train YOLOv11n on all three datasets and record performance metrics.

**Training Script Template:**

> **Important:** For each dataset (Dataset-10, Dataset-25, Dataset-50), you will train the model **3 times** with different epoch values: **10, 30, and 50 epochs**. This means you will have a total of **9 training runs** (3 datasets × 3 epoch values). Replace the `epochs` parameter in the script accordingly.

> Also replace `Dataset-<#>` with the actual dataset folder name (Dataset-10, Dataset-25, or Dataset-50) based on which dataset you are training.

Create a file named `train_model.py`:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data="Dataset-<#>/data.yaml",
    epochs=100,
    imgsz=640,
    device=0,
    workers=1,
    batch=8,
    patience=30,
    save=True,
    project="runs/detect",
    name="oscilloscope_jetson_<#>"
)

metrics = model.val()
model.export(format="engine")
```

**Usage Examples:**

- For Dataset-10 training, replace `Dataset-<#>` with `Dataset-10` and `oscilloscope_jetson_<#>` with `oscilloscope_jetson_10`
- For Dataset-25 training, replace `Dataset-<#>` with `Dataset-25` and `oscilloscope_jetson_<#>` with `oscilloscope_jetson_25`
- For Dataset-50 training, replace `Dataset-<#>` with `Dataset-50` and `oscilloscope_jetson_<#>` with `oscilloscope_jetson_50`

Then run the training:
```bash
python train_model.py
```

**Metrics to Record:**
- mAP@50 (mean Average Precision at 50% IoU)
- mAP@50-95 (mean Average Precision at 50-95% IoU)
- Precision
- Recall
- Training time (epochs)
- Inference speed (ms per image)

**Create a comparison table in your lab report:**

| Metric | Dataset_10 | Dataset_25 | Dataset_50 |
|--------|-----------|-----------|-----------|
| mAP@50 | | | |
| mAP@50-95 | | | |
| Precision | | | |
| Recall | | | |
| Training Time | | | |

---

****
