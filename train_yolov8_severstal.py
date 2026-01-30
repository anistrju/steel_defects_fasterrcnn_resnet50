# train_yolov8_severstal.py
"""
Training script for YOLOv8 segmentation on Severstal Steel Defect dataset.
Assumes:
- Dataset converted to YOLO format in folder: severstal_yolo/
- severstal.yaml exists and points correctly to train/val folders
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import argparse


def load_yaml_config(yaml_path: str) -> dict:
    """Load and validate the dataset YAML file."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {yaml_path}")

    with yaml_path.open('r') as f:
        config = yaml.safe_load(f)

    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in {yaml_path}")

    print("Dataset config loaded:")
    print(f"  Classes ({config['nc']}): {config['names']}")
    print(f"  Train: {config['train']}")
    print(f"  Val:   {config['val']}")
    return config


def train_model(
    model_variant: str = 'yolov8n-seg',
    data_yaml: str = 'severstal.yaml',
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = -1,           # -1 = auto (80% of GPU memory)
    device: str = None,
    project: str = 'runs/severstal',
    name: str = 'yolov8-seg-severstal',
    patience: int = 20,
    optimizer: str = 'auto',
    seed: int = 42,
    amp: bool = True,          # Automatic Mixed Precision
    cache: bool = False,       # Cache images in RAM (if you have enough memory)
):
    """
    Main training function with sensible defaults for Severstal.
    """
    print(f"Starting YOLOv8 training on Severstal dataset...")
    print(f"Model: {model_variant}")
    print(f"Data:  {data_yaml}")
    print(f"Epochs: {epochs} | Image size: {imgsz} | Batch: {batch}")

    # Load model (pretrained on COCO segmentation)
    model_name = f"{model_variant}.pt"
    try:
        model = YOLO(model_name)
        print(f"Loaded pretrained model: {model_name}")
    except Exception as e:
        print(f"Failed to load {model_name} → falling back to yolov8n-seg.pt")
        model = YOLO('yolov8n-seg.pt')

    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon
    print(f"Using device: {device}")

    # Train!
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            patience=patience,
            optimizer=optimizer,
            seed=seed,
            amp=amp,
            cache=args.cache,
            # Severstal-specific tuning suggestions
            lr0=0.001,              # lower starting lr often better for fine-tuning
            lrf=0.01,               # final OneCycleLR learning rate (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,                # box loss gain
            cls=0.5,                # cls loss gain
            dfl=1.5,                # distribution focal loss gain
            pose=0.0,               # usually 0 for detection/segmentation
            kobj=1.0,
            label_smoothing=0.0,
            nbs=64,                 # nominal batch size
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,             # save confusion matrix, PR curves, etc.
            save_period=10,         # save checkpoint every 10 epochs
            # Augmentations (good for industrial defects)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,            # no rotation for steel sheets
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
        )

        print("\nTraining finished!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")

        # Final validation
        print("\nRunning final validation on best model...")
        model.val()

        # Export formats useful for deployment
        print("\nExporting model...")
        model.export(format='torchscript')      # fast PyTorch inference
        model.export(format='onnx')             # for TensorRT / OpenVINO
        # model.export(format='engine')         # TensorRT (requires GPU + nvidia runtime)

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Severstal Steel Defects")
    parser.add_argument('--model', type=str, default='yolov8n-seg', help='YOLOv8 model variant (yolov8n-seg, yolov8s-seg, etc.)')
    parser.add_argument('--data', type=str, default='severstal.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=-1, help='Batch size (-1 = auto)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda, cpu, mps)')
    parser.add_argument('--project', type=str, default='runs/severstal', help='Project save directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--cache', action='store_true', help='Enable dataset caching (RAM)')

    args = parser.parse_args()

    train_model(
        model_variant=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )