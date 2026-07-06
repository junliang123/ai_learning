from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import CarSegmentationDataset
from model import UNet


BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 1
IMAGE_SIZE = (256, 256)
VAL_RATIO = 0.2
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"

def dice_coeff(pred, target, eps=1e-6):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    inter = pred*target

    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    inter_sum = inter.sum(dim=1)

    dice = (2*inter_sum + eps)/(pred_sum + target_sum + eps)
    return dice.mean()

def compute_dice_loss(pred, target):
    return 1 - dice_coeff(pred, target)

def train_one_epoch(model, loader, optimizer, bce_loss_fn, device, scaler):
    model.train()

    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.autocast(device_type=device.type, enabled=AMP):
            logits = model(images)
            bce_loss = bce_loss_fn(logits, masks)
            probs = torch.sigmoid(logits)
            dice_loss = compute_dice_loss(probs, masks)
            loss = bce_loss + dice_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item()*batch_size
        total_bce_loss += bce_loss.item()*batch_size
        total_dice_loss += dice_loss.item()*batch_size

    dataset_size = len(loader.dataset)
    avg_loss = total_loss/dataset_size
    avg_bce_loss = total_bce_loss/dataset_size
    avg_dice_loss = total_dice_loss/dataset_size
    return avg_loss, avg_bce_loss, avg_dice_loss

def evaluate(model, loader, device):
    model.eval()

    total_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            dice = dice_coeff(preds, masks)

            batch_size = images.size(0)
            total_dice += dice.item() * batch_size

    dataset_size = len(loader.dataset)
    avg_dice = total_dice / dataset_size

    return avg_dice

def main():
    root = Path(__file__).resolve().parent
    image_dir = root / "data" / "train_hq"
    mask_dir = root / "data" / "train_masks"
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    device = torch.device(DEVICE)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = CarSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        size=IMAGE_SIZE,
    )

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = UNet(NUM_CLASSES).to(device)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    best_dice = 0.0

    print(f"device: {device}")
    print(f"amp: {AMP}")
    print(f"dataset size: {len(dataset)}")
    print(f"train size: {len(train_dataset)}")
    print(f"val size: {len(val_dataset)}")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_bce_loss, train_dice_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            bce_loss_fn=bce_loss_fn,
            device=device,
            scaler=scaler,
        )
        val_dice = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )
        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"loss: {train_loss:.4f} "
            f"bce: {train_bce_loss:.4f} "
            f"dice_loss: {train_dice_loss:.4f} "
            f"val_dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "image_size": IMAGE_SIZE,
                "num_classes": NUM_CLASSES,
            }
            torch.save(checkpoint, checkpoint_dir / "best_unet.pth")
            print(f"Saved best model with dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
