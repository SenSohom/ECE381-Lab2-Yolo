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

---

## Step 1: System Update

Open a terminal and run the following commands to update the system:
```bash
sudo apt update
sudo apt upgrade
```

When prompted, enter the machine password:
```
jetson
```

> **Note:** The upgrade step may take a few minutes depending on the number of packages to update. Wait for it to complete fully before proceeding.
