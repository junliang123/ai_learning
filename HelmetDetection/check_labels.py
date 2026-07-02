from pathlib import Path
import random
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "yolo_dataset"

IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"
OUT_DIR = BASE_DIR / "label_check"

NAMES = ["helmet", "head", "person"]
COLORS = ["red", "blue", "green"]

OUT_DIR.mkdir(exist_ok=True)

image_files = list(IMAGE_DIR.glob("*.png"))

random.seed(42)
samples = random.sample(image_files, 10)

for img_path in samples:
    label_path = LABEL_DIR / f"{img_path.stem}.txt"

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            class_id, x_center, y_center, box_w, box_h = line.strip().split()
            class_id = int(class_id)
            x_center = float(x_center)
            y_center = float(y_center)
            box_w = float(box_w)
            box_h = float(box_h)

            xmin = (x_center - box_w / 2) * w
            ymin = (y_center - box_h / 2) * h
            xmax = (x_center + box_w / 2) * w
            ymax = (y_center + box_h / 2) * h

            color = COLORS[class_id]
            name = NAMES[class_id]

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
            draw.text((xmin, ymin), name, fill=color)

    img.save(OUT_DIR / img_path.name)