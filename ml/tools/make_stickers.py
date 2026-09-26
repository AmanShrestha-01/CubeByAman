"""Turn cube-face photos into single-sticker images.

Step 1 (you): for every photo, drag a box around the cube face.
Step 2 (computer): cut each box into a 3x3 grid, keep the middle of each
cell, and save the 9 stickers into the folder of the photo's colour.

Run from the ml/ folder:
    .venv/bin/python tools/make_stickers.py              # box new photos, then cut
    .venv/bin/python tools/make_stickers.py --redo IMG_7833.JPG   # redraw one box

Boxes are saved to data/boxes.json after every photo, so you can quit
(press c) and continue later without losing work.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2

PHOTOS = Path("data/raw/photos")
STICKERS = Path("data/stickers")
BOXES = Path("data/boxes.json")

SPLITS = ["train", "test_normal", "test_bad_light"]
COLOURS = ["white", "yellow", "red", "orange", "blue", "green"]

TRIM = 0.2         # cut 20% off every side of a cell -> keep the middle 60%
SIZE = 64          # every sticker image is saved as 64 x 64 pixels
SCREEN_H = 900     # photos are shrunk to this height so they fit on screen
WINDOW = "Draw a box around the cube face"


def all_photos():
    """Every photo as (key, path), e.g. ("train/red/IMG_7841.JPG", Path(...))."""
    for split in SPLITS:
        for colour in COLOURS:
            for path in sorted((PHOTOS / split / colour).glob("*.JPG")):
                yield f"{split}/{colour}/{path.name}", path


def draw_boxes(boxes):
    """Show each photo without a box and let the user drag one. False = user quit."""
    todo = [(k, p) for k, p in all_photos() if k not in boxes]
    for i, (key, path) in enumerate(todo, start=1):
        img = cv2.imread(str(path))
        scale = SCREEN_H / img.shape[0]
        small = cv2.resize(img, None, fx=scale, fy=scale)
        print(f"[{i}/{len(todo)}] {key}   drag box, ENTER = next, c = quit")

        x, y, w, h = cv2.selectROI(WINDOW, small, showCrosshair=True)
        if w == 0 or h == 0:                      # pressed c (or no box drawn)
            cv2.destroyAllWindows()
            return False

        # convert the box from the shrunken picture back to the full-size photo
        boxes[key] = [round(v / scale) for v in (x, y, w, h)]
        BOXES.write_text(json.dumps(boxes, indent=1))
    cv2.destroyAllWindows()
    return True


def cut_face(img, box):
    """Split the boxed face into 3x3 cells; return [(row, col, 64x64 sticker)]."""
    x, y, w, h = box
    face = img[y:y + h, x:x + w]
    stickers = []
    for row in range(3):
        for col in range(3):
            # each cell is 1/3 of the face; skip TRIM of it on every side
            y0, y1 = int((row + TRIM) * h / 3), int((row + 1 - TRIM) * h / 3)
            x0, x1 = int((col + TRIM) * w / 3), int((col + 1 - TRIM) * w / 3)
            cell = face[y0:y1, x0:x1]
            stickers.append((row, col, cv2.resize(cell, (SIZE, SIZE), interpolation=cv2.INTER_AREA)))
    return stickers


def cut_all(boxes):
    """Rebuild data/stickers/ from scratch using every saved box."""
    shutil.rmtree(STICKERS, ignore_errors=True)
    count = 0
    for key, path in all_photos():
        if key not in boxes:
            continue
        split, colour, _ = key.split("/")
        out_dir = STICKERS / split / colour
        out_dir.mkdir(parents=True, exist_ok=True)
        for row, col, sticker in cut_face(cv2.imread(str(path)), boxes[key]):
            if colour == "white" and (row, col) == (1, 1):
                continue                          # white centre has a logo, not plain white
            cv2.imwrite(str(out_dir / f"{path.stem}_r{row}c{col}.png"), sticker)
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", metavar="PHOTO", help="forget the box for this photo and draw it again")
    args = parser.parse_args()

    boxes = json.loads(BOXES.read_text()) if BOXES.exists() else {}
    if args.redo:
        boxes = {k: v for k, v in boxes.items() if not k.endswith("/" + args.redo)}

    finished = draw_boxes(boxes)
    total = sum(1 for _ in all_photos())
    print(f"\nBoxes saved: {len(boxes)}/{total}")
    print(f"Stickers written: {cut_all(boxes)} -> {STICKERS}/")
    if not finished:
        print("Stopped early. Run the same command again to continue where you left off.")
