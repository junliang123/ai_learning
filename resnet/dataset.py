from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

class HerbDataset(Dataset):
    def __init__(self, txt_path, image_root, transform=None):
        self.txt_path = Path(txt_path)
        self.image_root = Path(image_root)
        self.transform = transform
        self.samples = []

        with open(self.txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                img_rel_path, label = line.rsplit(",", 1)
                self.samples.append((img_rel_path, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]
        img_path = self.image_root / img_rel_path
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    train_dataset = HerbDataset(
        txt_path="data/split/train.txt",
        image_root="data/raw/train",
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])