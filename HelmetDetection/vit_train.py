import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms

from vit_model import VIT
from dataset import HelmetDetectionDataset, detection_collate_fn

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = HelmetDetectionDataset(
    image_dir="yolo_dataset/images/train",
    label_dir="yolo_dataset/labels/train",
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=detection_collate_fn,
)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_dataset = HelmetDetectionDataset(
    image_dir="yolo_dataset/images/val",
    label_dir="yolo_dataset/labels/val",
    transform=val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=detection_collate_fn,
)

model = VIT().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def box_cxcywh_to_xyxy(boxes):
    x_center, y_center, w, h = boxes.unbind(-1)
    x_min = x_center - 0.5 * w
    y_min = y_center - 0.5 * h
    x_max = x_center + 0.5 * w
    y_max = y_center + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

def box_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    area1 = (boxes1[:, 2] - boxes1[:, 0])*(boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0])*(boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou

def compute_loss(class_logits, bbox_preds, targets):
    batch_size = class_logits.size(0)
    num_predictions = class_logits.size(1)
    num_classes_with_bg = class_logits.size(2)
    device = class_logits.device
    no_object_id = NUM_CLASSES

    total_bbox_loss = torch.tensor(0.0, device=device)
    total_matches = 0
    target_classes = torch.full(
        (batch_size, num_predictions),
        fill_value=no_object_id,
        dtype=torch.long,
        device=device,
    )

    for b in range(batch_size):
        pred_logits = class_logits[b]
        pred_probs = pred_logits.softmax(-1)
        pred_boxes = bbox_preds[b]

        target_boxes = targets[b]["boxes"].to(device)
        target_labels = targets[b]["labels"].to(device)

        num_targets = target_boxes.size(0)
        if num_targets == 0:
            continue

        iou_cost = 1 - box_iou(pred_boxes, target_boxes)

        target_probs = torch.zeros(
            (num_targets, pred_probs.size(-1)),
            device=device,
        )
        target_probs[
            torch.arange(num_targets, device=device),
            target_labels.long(),
        ] = 1

        cls_cost = torch.cdist(pred_probs, target_probs, p=1)

        cost_matrix = iou_cost + cls_cost

        row_ind, col_ind = linear_sum_assignment(
            cost_matrix.detach().cpu().numpy()
        )

        row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
        col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)

        matched_pred_boxes = pred_boxes[row_ind]
        matched_target_boxes = target_boxes[col_ind]
        matched_target_labels = target_labels[col_ind]

        target_classes[b, row_ind] = matched_target_labels.long()

        bbox_loss = F.l1_loss(
            matched_pred_boxes,
            matched_target_boxes,
            reduction="sum",
        )

        total_bbox_loss += bbox_loss
        total_matches += row_ind.numel()

    total_matches = max(total_matches, 1)

    total_bbox_loss = total_bbox_loss / total_matches

    class_weights = torch.ones(num_classes_with_bg, device=device)
    class_weights[no_object_id] = 0.1
    total_class_loss = F.cross_entropy(
        class_logits.reshape(-1, num_classes_with_bg),
        target_classes.reshape(-1),
        weight=class_weights,
    )

    total_loss = total_class_loss + total_bbox_loss

    return total_loss, total_class_loss, total_bbox_loss

def postprocess_predictions(class_logits, bbox_preds, confidence_threshold=0.1, iou_threshold=0.5):
    results = []

    batch_size = class_logits.size(0)

    for b in range(batch_size):
        pred_logits = class_logits[b]
        pred_boxes = bbox_preds[b]

        pred_probs = pred_logits.softmax(dim=-1)

        foreground_probs = pred_probs[:, :NUM_CLASSES]
        confidence, predicted_class = foreground_probs.max(dim=-1)

        keep = confidence > confidence_threshold

        filtered_boxes = pred_boxes[keep]
        filtered_classes = predicted_class[keep]
        filtered_confidences = confidence[keep]

        if filtered_boxes.size(0) == 0:
            results.append({
                "boxes": filtered_boxes,
                "classes": filtered_classes,
                "scores": filtered_confidences,
            })
            continue

        filtered_boxes_xyxy = box_cxcywh_to_xyxy(filtered_boxes)
        filtered_boxes_xyxy = filtered_boxes_xyxy.clamp(0, 1)

        keep_indices = nms(
            filtered_boxes_xyxy,
            filtered_confidences,
            iou_threshold,
        )

        final_boxes = filtered_boxes[keep_indices]
        final_classes = filtered_classes[keep_indices]
        final_confidences = filtered_confidences[keep_indices]

        results.append({
            "boxes": final_boxes,
            "classes": final_classes,
            "scores": final_confidences,
        })

    return results

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    running_loss = 0.0
    running_class_loss = 0.0
    running_bbox_loss = 0.0
    total_images = 0

    for images, targets in loader:
        images = images.to(device)
        class_logits, bbox_preds = model(images)

        loss, class_loss, bbox_loss = compute_loss(
            class_logits,
            bbox_preds,
            targets,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        running_class_loss += class_loss.item() * batch_size
        running_bbox_loss += bbox_loss.item() * batch_size
        total_images += batch_size

    epoch_loss = running_loss / total_images
    epoch_class_loss = running_class_loss / total_images
    epoch_bbox_loss = running_bbox_loss / total_images

    return epoch_loss, epoch_class_loss, epoch_bbox_loss

def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    running_class_loss = 0.0
    running_bbox_loss = 0.0
    total_images = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            class_logits, bbox_preds = model(images)

            loss, class_loss, bbox_loss = compute_loss(
                class_logits,
                bbox_preds,
                targets,
            )

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_class_loss += class_loss.item() * batch_size
            running_bbox_loss += bbox_loss.item() * batch_size
            total_images += batch_size

    epoch_loss = running_loss / total_images
    epoch_class_loss = running_class_loss / total_images
    epoch_bbox_loss = running_bbox_loss / total_images

    return epoch_loss, epoch_class_loss, epoch_bbox_loss

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    train_loss, train_class_loss, train_bbox_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=DEVICE,
    )
    val_loss, val_class_loss, val_bbox_loss = evaluate(
        model=model,
        loader=val_loader,
        device=DEVICE,
    )
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"train_loss: {train_loss:.4f} "
        f"train_cls: {train_class_loss:.4f} "
        f"train_bbox: {train_bbox_loss:.4f} "
        f"val_loss: {val_loss:.4f} "
        f"val_cls: {val_class_loss:.4f} "
        f"val_bbox: {val_bbox_loss:.4f}"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "vit_detector_best.pth")
