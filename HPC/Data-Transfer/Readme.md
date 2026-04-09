# Transferring the Dataset from Google Drive to Wulver

The training dataset is shared via Google Drive. Follow one of the two methods below to get it onto Wulver.

---

## Method 1 — `gdown` (Recommended)

Use this when the instructor has shared the Google Drive folder with **"Anyone with the link"**.

### Step 1 — Navigate to your course directory and activate your environment

```bash
cd /course/2026/spring/ece/381/th36/$LOGNAME
module load Miniforge3/24.11.3-0
conda activate fine-tune
```

### Step 2 — Install gdown

```bash
pip install gdown
```

### Step 3 — Download the dataset

```bash
gdown --folder "https://drive.google.com/drive/folders/FOLDER_ID" -O Dataset/
```

Replace `FOLDER_ID` with the ID from the shared Google Drive URL:

```
https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9I0J
                                        ^^^^^^^^^^^^^^^^^^^^
                                        this part is the ID
```

> If the download is interrupted, re-run the same command with `--remaining-ok` to resume:
> ```bash
> gdown --folder "https://drive.google.com/drive/folders/FOLDER_ID" -O Dataset/ --remaining-ok
> ```

---

## Method 2 — Download Locally then `scp` to Wulver

Use this if `gdown` does not work or the folder requires Google account login.

### Step 1 — Download the dataset to your local machine

1. Open the shared Google Drive link in your browser
2. Right-click the dataset folder → **Download**
3. Google Drive will zip the folder — save the `.zip` file to your local machine

### Step 2 — Transfer to Wulver using `scp`

Open a terminal on your local machine (not on Wulver) and run:

**Mac / Linux:**
```bash
scp -r /path/to/Dataset/ UCID@wulver.njit.edu:/course/2026/spring/ece/381/th36/UCID/
```

**Windows (in MobaXterm terminal):**
```bash
scp -r C:/Users/YourName/Downloads/Dataset/ UCID@wulver.njit.edu:/course/2026/spring/ece/381/th36/UCID/
```

Replace `UCID` with your actual UCID in both the command and the destination path.

### Step 3 — Unzip if needed

If the downloaded file is a `.zip`, unzip it on Wulver:

```bash
cd /course/2026/spring/ece/381/th36/$LOGNAME
unzip Dataset.zip -d Dataset/
```

---

## After Transfer — Verify the Structure

Once the transfer is complete, confirm the dataset looks correct:

```bash
ls /course/2026/spring/ece/381/th36/$LOGNAME/Dataset/
```

You should see:

```
Shot-1/  Shot-2/  Shot-3/  Shot-4/  Shot-5/
```

Then update `DATASET_ROOT` in `prepare_dataset.py` to point to this folder:

```python
DATASET_ROOT = "/course/2026/spring/ece/381/th36/UCID/Dataset"
```
