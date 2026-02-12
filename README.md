# Lab-2 : YOLO Manual

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

#### 5.1: Install Ninja Build System

```bash
apt install ninja-build
```

#### 5.2: Clone OpenCV Repository

```bash
cd /tmp
git clone --depth 1 https://github.com/opencv/opencv.git
```

#### 5.3: Navigate to OpenCV Directory and Create Build Folder

```bash
cd opencv
mkdir build && cd build
```

#### 5.4: Configure OpenCV with CMake

```bash
cmake -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_GTK=ON \
  ..
```

#### 5.5: Compile OpenCV with Ninja

```bash
ninja -j$(nproc)
```

> **Note:** This step will take some time as it compiles OpenCV with all available CPU cores. The `$(nproc)` command automatically detects the number of available processor cores.

#### 5.6: Install OpenCV

```bash
ninja install
```

#### 5.7: Update Library Cache

```bash
ldconfig
```

> **Note:** This command updates the system's library cache to recognize the newly installed OpenCV libraries.

---

### Step 6: Save Docker Changes and Create Custom Image

**Objective:** Save all the changes made to the Docker container (OpenCV, display libraries, and dependencies) as a new custom Docker image for future use.

#### 6.1: Open a New Terminal

Open a **new terminal window** on your Jetson Orin Nano (do not close the current Docker container terminal yet).

#### 6.2: List Active Docker Containers

Run the following command to see all active Docker containers:

```bash
sudo docker ps -a
```

This will return a list of Docker containers. The **first entry** is your latest container where all the changes have been made. **Copy the Container ID** from the output.

#### 6.3: Commit the Container to Create a New Image

Execute the following command, replacing `<Container-ID>` with the ID you copied:

```bash
sudo docker commit <Container-ID> ultralytics-opencv:jetson-jetpack6
```

**Example:** If your Container ID is `a1b2c3d4e5f6`, the command would be:
```bash
sudo docker commit a1b2c3d4e5f6 ultralytics-opencv:jetson-jetpack6
```

Once the command executes successfully, you will see a response confirming the commit.

#### 6.4: Close Both Terminals

After the commit is complete, close both the Docker container terminal and the new terminal you just opened.

#### 6.5: Use the Custom Image for Future Sessions

From now on, you can skip all the installation steps (Steps 3, 4, and 5) and directly launch the Docker container with your custom image using:

```bash
sudo docker run -it --ipc=host --runtime=nvidia --device=/dev/video0 \
  -v /home/ece381-<kit#>/Documents/Lab2-Workspace:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  ultralytics-opencv:jetson-jetpack6 bash
```

> **Note:** Replace `<kit#>` with your kit number (same as before).

This custom image includes all the necessary libraries (OpenCV with GTK support, display libraries, and build dependencies), so you won't need to reinstall them in future sessions.

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

**Create a file named `ObjectDetect.py`:**

```python
import cv2
from ultralytics import YOLO
import os

# Load YOLO model using TensorRT engine
model = YOLO("yolo11n.engine", task="detect")

output_folder = "processed_frames"
os.makedirs(output_folder, exist_ok=True)

frame_counter = 0

# Open webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Process and visualize results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]} {conf:.2f}"  # Class label with confidence

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Display the frame
    frame_filename = os.path.join(output_folder, f"frame_{frame_counter:04d}.jpg")
    cv2.imshow("YOLOv11n Object Detection", frame)
    #cv2.imwrite(frame_filename, frame)
    frame_counter += 1

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```
**Run the script:**
```bash
python ObjectDetect.py
```

This will:
- Load the YOLOv11n TensorRT engine model
- Capture video from your webcam in real-time
- Display object detection with bounding boxes and confidence scores
- Press 'q' to exit the application

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

**Refer to the video ["Annotate-using-Roboflow.mp4"](https://drive.google.com/file/d/1f_iMva0YXieoD2cwlBEcx_3YUIdhVSjy/view?usp=sharing)** to learn how to use the tool and download the custom dataset.

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

> **Important:** For each dataset (Dataset-10, Dataset-25, Dataset-50), you will train the model **3 times** with different epoch values: **25, 50, and 100 epochs**. This means you will have a total of **9 training runs** (3 datasets × 3 epoch values). Replace the `epochs` parameter in the script accordingly.

> Also replace `Dataset-<#>` with the actual dataset folder name (Dataset-10, Dataset-25, or Dataset-50) based on which dataset you are training. To create the python file follow the steps mentioned in your ppt. Then copy paste this code. 

Create a file named `train_model.py`:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data="Dataset-<#>/data.yaml",
    epochs=100,
    imgsz=640,
    device=0,
    workers=0,
    batch=4,
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

Then run the training. Make sure to close all the other applications before you start the training process, it includes your browsers, text editors etc, only your terminal should be open and nothing else:
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

All these things will be printed in the terminal during the training phase for each epoch. Once the training is finished you can copy the metrics from the terminal, paste it in a txt file and then use the help of any AI tool to plot the curves and attach to your Lab Report.

**Create a comparison table in your lab report:**

| Metric | Dataset_10 | Dataset_25 | Dataset_50 |
|--------|-----------|-----------|-----------|
| mAP@50 | | | |
| mAP@50-95 | | | |
| Precision | | | |
| Recall | | | |
| Training Time | | | |

And all the Curves generated.

---
### Task 3.4: Inference on Trained Models
**Objective:** Test the trained models on new images and record detection results.

**Inference Script:**

Create a file named `inference.py`:

```python
import cv2
from ultralytics import YOLO
import os

# Load YOLO model using TensorRT engine
model = YOLO("/ultralytics/runs/detect/runs/detect/oscilloscope_jetson_<#>/weights/best.engine", task="detect")

output_folder = "processed_frames"
os.makedirs(output_folder, exist_ok=True)

frame_counter = 0

# Open webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Process and visualize results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]} {conf:.2f}"  # Class label with confidence

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Display the frame
    frame_filename = os.path.join(output_folder, f"frame_{frame_counter:04d}.jpg")
    cv2.imshow("YOLOv11n Object Detection", frame)
    #cv2.imwrite(frame_filename, frame)
    frame_counter += 1

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

**Instructions:**

1. Update the model path with your trained model (e.g., `oscilloscope_jetson_10`, `oscilloscope_jetson_25`, or `oscilloscope_jetson_50`)

2. Run the inference script:
```bash
python inference.py
```

3. The script will open a webcam feed with real-time detections showing:
   - Bounding boxes around detected objects
   - Object class labels (Oscilloscope or Jetson)
   - Confidence scores for each detection

4. **Capture Screenshots:**
   - Take at least **5 screenshots** for each trained model showing successful detections
   - Make sure the screenshots clearly show:
     - Object class names
     - Confidence scores
     - Bounding boxes
   - Save screenshots with filenames: `inference_model_10_01.png`, `inference_model_10_02.png`, etc.

5. **Include all screenshots in your lab report** with annotations explaining the detections and comparing performance across different dataset sizes.

**Example Screenshot Naming Convention:**
- `inference_model_10_01.png` - First screenshot from 10-image dataset model
- `inference_model_25_03.png` - Third screenshot from 25-image dataset model
- `inference_model_50_02.png` - Second screenshot from 50-image dataset model

---

## Lab Report Due Date : 20 Feb 2026
