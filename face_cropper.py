"""
face_cropper.py
---------------
Scans an input directory for image files, detects faces using OpenCV,
crops each face (with optional padding), and saves the results to an
output directory.

Usage:
    python face_cropper.py --input ./photos --output ./faces
    python face_cropper.py --input ./photos --output ./faces --padding 0.3 --min-size 60
"""

import argparse
import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Haar cascade shipped with OpenCV – no download required
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_detector() -> cv2.CascadeClassifier:
    """Load the Haar cascade face detector."""
    detector = cv2.CascadeClassifier(HAAR_CASCADE)
    if detector.empty():
        sys.exit("ERROR: Could not load face detector cascade. Is OpenCV installed correctly?")
    return detector


def detect_faces(gray_img, detector: cv2.CascadeClassifier, min_size: int):
    """
    Return a list of (x, y, w, h) bounding boxes for detected faces.

    scaleFactor  – how much the image is reduced at each scale level.
    minNeighbors – higher = fewer detections but higher quality.
    minSize      – minimum face size in pixels (width × height).
    """
    return detector.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )


def crop_face(image, x: int, y: int, w: int, h: int, padding: float):
    """
    Crop a face region from *image* with proportional padding on all sides.

    padding = 0.2 means add 20 % of the face width/height as a border.
    The crop is clamped to the image boundaries.
    """
    img_h, img_w = image.shape[:2]
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return image[y1:y2, x1:x2]


def process_directory(
    input_dir: Path,
    output_dir: Path,
    padding: float,
    min_size: int,
    overwrite: bool,
) -> dict:
    """
    Walk *input_dir*, detect faces in every supported image, and write crops
    to *output_dir*.

    Returns a summary dict with counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = load_detector()

    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        print(f"No supported image files found in '{input_dir}'.")
        return {"images": 0, "faces_saved": 0, "skipped": 0, "errors": 0}

    images_processed = 0
    faces_saved = 0
    skipped = 0
    errors = 0

    print(f"Found {len(image_paths)} image(s) in '{input_dir}'.\n")

    for img_path in sorted(image_paths):
        print(f"  Processing: {img_path.name}")

        # --- Load image ---
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    ✗ Could not read file – skipping.")
            errors += 1
            continue

        images_processed += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improve detection under varied lighting

        # --- Detect faces ---
        faces = detect_faces(gray, detector, min_size)

        if len(faces) == 0:
            print(f"    – No faces detected.")
            skipped += 1
            continue

        print(f"    ✓ {len(faces)} face(s) detected.")

        # --- Save each face crop ---
        stem = img_path.stem
        suffix = img_path.suffix.lower()

        for idx, (x, y, w, h) in enumerate(faces, start=1):
            # Build output filename: original_stem_face1.jpg, etc.
            face_filename = f"{stem}_face{idx}{suffix}"
            out_path = output_dir / face_filename

            if out_path.exists() and not overwrite:
                print(f"    – '{face_filename}' already exists, skipping (use --overwrite to replace).")
                skipped += 1
                continue

            crop = crop_face(image, x, y, w, h, padding)

            success = cv2.imwrite(str(out_path), crop)
            if success:
                h_px, w_px = crop.shape[:2]
                print(f"    → Saved '{face_filename}' ({w_px}×{h_px} px)")
                faces_saved += 1
            else:
                print(f"    ✗ Failed to write '{face_filename}'.")
                errors += 1

    return {
        "images": images_processed,
        "faces_saved": faces_saved,
        "skipped": skipped,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop faces from images in a directory and save them to another directory."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Path to the directory containing source images.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        type=Path,
        help="Path to the directory where cropped face images will be saved.",
    )
    parser.add_argument(
        "--padding", "-p",
        type=float,
        default=0.2,
        help="Fractional padding around each detected face (default: 0.2 = 20%%).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=60,
        help="Minimum face size in pixels to detect (default: 60).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.is_dir():
        sys.exit(f"ERROR: Input path '{args.input}' is not a directory or does not exist.")

    if not (0.0 <= args.padding <= 1.0):
        sys.exit("ERROR: --padding must be between 0.0 and 1.0.")

    print("=" * 50)
    print("  Face Cropper")
    print("=" * 50)
    print(f"  Input  : {args.input.resolve()}")
    print(f"  Output : {args.output.resolve()}")
    print(f"  Padding: {args.padding * 100:.0f}%")
    print(f"  Min sz : {args.min_size}px")
    print("=" * 50 + "\n")

    summary = process_directory(
        input_dir=args.input,
        output_dir=args.output,
        padding=args.padding,
        min_size=args.min_size,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    print(f"  Images processed : {summary['images']}")
    print(f"  Faces saved      : {summary['faces_saved']}")
    print(f"  Skipped          : {summary['skipped']}")
    print(f"  Errors           : {summary['errors']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
