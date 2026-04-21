# Face Cropper

A Python script that scans a directory of images, detects faces using OpenCV's
Haar cascade classifier, and saves individual cropped face images to a separate
output directory.

---

## Setup

**1. Create and activate a virtual environment:**

```bash
# Create the venv
python -m venv venv

# Activate it — macOS/Linux:
source venv/bin/activate

# Activate it — Windows:
venv\Scripts\activate
```

**2. Install dependencies** (OpenCV is the only requirement):

```bash
pip install -r requirements.txt
```

> To deactivate the virtual environment when you're done, just run `deactivate`.

---

## Usage

```bash
python face_cropper.py --input ./photos --output ./faces
```

### All options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input`    | `-i` | *(required)* | Directory containing source images |
| `--output`   | `-o` | *(required)* | Directory where face crops are saved |
| `--padding`  | `-p` | `0.2`        | Padding around each face as a fraction of face size (0.0–1.0) |
| `--min-size` |      | `60`         | Minimum face width/height in pixels to detect |
| `--overwrite`|      | `false`      | Overwrite existing output files instead of skipping |

### Examples

```bash
# Basic run
python face_cropper.py -i ./photos -o ./faces

# More padding (30%) and overwrite existing crops
python face_cropper.py -i ./photos -o ./faces --padding 0.3 --overwrite

# Stricter detection – only detect larger faces (100px+)
python face_cropper.py -i ./photos -o ./faces --min-size 100
```

---

## How it works

1. **Scan** — Lists all `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp` files in the input directory.
2. **Detect** — Converts each image to grayscale, applies histogram equalization for better detection under varied lighting, then runs OpenCV's frontal face Haar cascade.
3. **Crop** — Each detected face bounding box is expanded by the padding percentage and clamped to image boundaries.
4. **Save** — Crops are written to the output directory as `<original_name>_face1.jpg`, `<original_name>_face2.jpg`, etc.

---

## Output naming

If `photos/team.jpg` contains 2 faces, the output will be:

```
faces/
  team_face1.jpg
  team_face2.jpg
```

---

## Limitations & tips

- **Front-facing only** — The Haar cascade works best with faces looking toward the camera. Profile or heavily angled faces may be missed.
- **Lighting matters** — Histogram equalization helps, but extremely dark or blown-out images will reduce accuracy.
- **False positives** — If non-face objects are being detected, increase `--min-size` or the `minNeighbors` value inside `detect_faces()`.
- **False negatives** — If real faces are being missed, decrease `--min-size` or try reducing `minNeighbors` to `4`.

---

## Extending the project

- **Deep learning detector** — Replace the Haar cascade with `cv2.dnn` + a pre-trained SSD or MTCNN model for significantly better accuracy on tilted/partial faces.
- **Batch subdirectories** — Change `input_dir.iterdir()` to `input_dir.rglob("*")` to recurse into subdirectories.
- **Resize output** — Add a `--size 128` flag and call `cv2.resize(crop, (size, size))` before saving for uniform thumbnails.
