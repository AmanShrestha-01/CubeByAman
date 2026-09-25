"""Collect training photos of cube stickers from the webcam.

Controls (click the video window first so it receives key presses):
    1-6    choose the colour you are about to photograph
    space  save the current frame to ml/data/raw/<colour>/
    q/Esc  quit

Run from anywhere:
    python ml/capture.py            # default camera
    python ml/capture.py --camera 1 # try another camera if 0 is wrong
"""

import argparse
import time
from pathlib import Path

import cv2

# Key '1' -> index 0, etc. Order is arbitrary but must stay fixed once you
# start collecting, because the folder names become the class labels.
COLOURS = ["white", "yellow", "red", "orange", "blue", "green"]

# OpenCV stores pixels as BGR (blue, green, red), not RGB.
LABEL_BGR = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "blue": (255, 0, 0),
    "green": (0, 200, 0),
}

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"


def count_images(folder: Path) -> int:
    """Existing photos, so counts carry over between sessions."""
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".png"})


def draw_text(img, text, org, colour, scale=0.8):
    """putText with a dark box behind it so it stays readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 5, y - h - 5), (x + w + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, colour, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. On macOS, allow camera access for "
            "your terminal / VS Code in System Settings > Privacy & Security > Camera."
        )

    counts = {c: count_images(RAW_DIR / c) for c in COLOURS}
    current = COLOURS[0]
    flash_until = 0.0  # briefly show "Saved" after each photo

    print("Keys: 1-6 choose colour | space save | q quit")
    print("Starting counts:", counts)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read a frame from the camera; stopping.")
                break

            # Draw on a copy so the saved photo has no text on it.
            display = frame.copy()
            draw_text(display, f"Colour: {current}", (15, 35), LABEL_BGR[current], 1.0)
            draw_text(display, f"Photos: {counts[current]}", (15, 75), (255, 255, 255))
            keys = "  ".join(f"{i + 1}:{c}" for i, c in enumerate(COLOURS))
            draw_text(display, keys, (15, display.shape[0] - 45), (200, 200, 200), 0.55)
            draw_text(display, "space: save   q: quit", (15, display.shape[0] - 15), (200, 200, 200), 0.55)
            if time.time() < flash_until:
                draw_text(display, "Saved", (display.shape[1] - 120, 35), (0, 255, 0))

            cv2.imshow("capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # 27 = Esc
                break
            elif ord("1") <= key <= ord("6"):
                current = COLOURS[key - ord("1")]
            elif key == ord(" "):
                folder = RAW_DIR / current
                folder.mkdir(parents=True, exist_ok=True)
                # Millisecond timestamp: unique, sortable, never overwrites.
                path = folder / f"{current}_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(path), frame)
                counts[current] += 1
                flash_until = time.time() + 0.4
                print(f"saved {path.relative_to(RAW_DIR.parent.parent)}  ({current}: {counts[current]})")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Final counts:", counts)


if __name__ == "__main__":
    main()
