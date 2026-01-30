import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2 as T
from PIL import Image
import os


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

NUM_CLASSES = 5   # background + 4 defects
SCORE_THRESHOLD = 0.01

WEIGHTS_PATH = "fasterrcnn_severstal.pth"


CLASS_MAP = {
    1: "Defect_1",
    2: "Defect_2",
    3: "Defect_3",
    4: "Defect_4",
}


# ─────────────────────────────────────────────────────────────
# Transforms (NO augmentation)
# ─────────────────────────────────────────────────────────────

def get_inference_transform():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((800, 800), antialias=True),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────

def get_model(num_classes=NUM_CLASSES):
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=ResNet50_Weights.IMAGENET1K_V1,
        trainable_layers=0
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes
    )

    return model


def load_model(weights_path, device):
    model = get_model()
    # Explicitly allow full pickle loading (safe here because it's your own file)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Optional but recommended: handle common "module." prefix from DataParallel training
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)  # strict=True → fails if keys don't match
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def run_inference(image_path, model, device, score_threshold=SCORE_THRESHOLD):
    transform = get_inference_transform()

    img = Image.open(image_path).convert("RGB")
    img_tensor, _ = transform(img, {})

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    keep = output["scores"] >= score_threshold

    boxes = output["boxes"][keep].cpu()
    labels = output["labels"][keep].cpu()
    scores = output["scores"][keep].cpu()

    return boxes, labels, scores


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(WEIGHTS_PATH, device)
    total_images = 0
    images_with_detections = 0

    image_dir = "test_images"

    for file_name in sorted(os.listdir(image_dir)):
        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, file_name)
        total_images += 1



        boxes, labels, scores = run_inference(
            image_path,
            model,
            device
        )

        print(f"\nImage: {file_name}")

        if len(boxes) == 0:
            print("  No defects detected")
            continue

        images_with_detections += 1
        for box, label, score in zip(boxes, labels, scores):
            class_name = CLASS_MAP.get(label.item(), "Unknown")
            
            print(
                f"  Class: {class_name:<10} "
                f"| Score: {score:.3f} "
                f"| Box: {[round(v, 1) for v in box.tolist()]}"
            )
    print(f"\nTotal images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")


if __name__ == "__main__":
    main()
