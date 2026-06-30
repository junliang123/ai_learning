from pathlib import Path

train_dir = Path("data/raw/train")
class_names = [
    p.name
    for p in train_dir.iterdir()
    if p.is_dir()
]
class_names = sorted(class_names)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
SPLIT_DIR = BASE_DIR / "data" / "split"

train_dir = RAW_DIR / "train"
with open(SPLIT_DIR / "train.txt", "w", encoding="utf-8") as f:
    for idx, name in enumerate(class_names):
        class_dir = train_dir / name
        for img_path in class_dir.glob("*.jpg"):
            relative_path = img_path.relative_to(train_dir)
            f.write(f"{relative_path.as_posix()},{idx}\n")

val_dir = RAW_DIR / "val"
with open(SPLIT_DIR / "val.txt", "w", encoding="utf-8") as f:
    for idx, name in enumerate(class_names):
        class_dir = val_dir / name
        for img_path in class_dir.glob("*.jpg"):
            relative_path = img_path.relative_to(val_dir)
            f.write(f"{relative_path.as_posix()},{idx}\n")

with open(SPLIT_DIR / "class_to_idx.txt", "w", encoding="utf-8") as f:
    for idx, name in enumerate(class_names):
        f.write(f"{name},{idx}\n")