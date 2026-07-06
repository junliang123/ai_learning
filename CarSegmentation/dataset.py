from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CarSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.size = size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()
            ])
        else: self.transform = transform
        self.image_path = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        mask_name = image_path.stem + "_mask.gif"
        mask_path = self.mask_dir / mask_name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = mask.resize(self.size, resample=Image.NEAREST)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0).float()
        return image, mask