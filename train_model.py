import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
from pycocotools.coco import COCO
from torch.optim.lr_scheduler import StepLR

class DefectDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        print(f"Loading index {index}")
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info['file_name'])
        print("Image path:", path)

        img = Image.open(path).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor([(ann['bbox'][2] * ann['bbox'][3]) for ann in anns], dtype=torch.float32),
            "iscrowd": torch.zeros((len(anns),), dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train):
    transforms = [
        T.ToImage(),                               # ✅ convert PIL → Tensor FIRST
        T.ToDtype(torch.float32, scale=True),
    ]

    if train:
        transforms += [
            T.RandomHorizontalFlip(0.5),
            T.RandomPhotometricDistort(),          # ✅ now works correctly
            T.RandomZoomOut(fill=0),
        ]

    transforms += [
        T.Resize((800, 800), antialias=True),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]

    return T.Compose(transforms)


def get_model(num_classes=5, backbone_name='resnet50'):
    # Option A: Use modern API with explicit backbone weights (recommended)
    if backbone_name == 'resnet50':
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1   # or .DEFAULT
        # You can also do ResNet50_Weights.IMAGENET1K_V2 if you have newer torchvision
    elif backbone_name == 'resnet101':
        from torchvision.models import resnet101, ResNet101_Weights
        backbone_weights = ResNet101_Weights.IMAGENET1K_V1
    else:
        raise ValueError("Only resnet50 / resnet101 supported here")

    # This creates ResNet → FPN backbone (same as used in fasterrcnn_resnet50_fpn_v2)
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=backbone_weights,           # ← uses your local .pth if already downloaded
        trainable_layers=3                  # usually 3–5; 0 = freeze all
    )

    # Now build the full Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,            # 4 defects + background
        # You can tune these if needed (defaults are usually fine)
        # rpn_pre_nms_top_n_train=2000,
        # box_detections_per_img=300,
    )

    return model


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # ─── Update these paths ───────────────────────────────────────
    root_dir = "severstal_yolo/train_images"
    ann_file = "severstal_yolo/instances_train.json"
    # ──────────────────────────────────────────────────────────────

    dataset = DefectDataset(root_dir, ann_file, get_transform(train=True))
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0 if device.type == 'cpu' else 4,   # 0 on CPU avoids issues
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = get_model(num_classes=5, backbone_name='resnet50')
    #model = get_model(num_classes=5)  # 4 defects + background
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()

            # Gradient clipping (helps stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(model.state_dict(), "fasterrcnn_severstal.pth")
    print("Training finished. Model saved.")


if __name__ == "__main__":
    main()