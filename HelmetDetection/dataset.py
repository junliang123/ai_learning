from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HelmetDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.image_paths = sorted(self.image_dir.glob("*.png"))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])  
                box_w = float(parts[3])
                box_h = float(parts[4])

                labels.append(class_id)
                boxes.append([x_center, y_center, box_w, box_h])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target
    
def detection_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = list(targets)
    return images, targets