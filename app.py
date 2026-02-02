import streamlit as st
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2 as T
from PIL import Image
import io
import pandas as pd
from typing import List, Dict

# ─────────────────────────────────────────────────────────────
# Configuration (from your script)
# ─────────────────────────────────────────────────────────────
NUM_CLASSES = 5
SCORE_THRESHOLD = 0.01
WEIGHTS_PATH = "/app/fasterrcnn_severstal.pth"  # Adjust if mounted elsewhere

CLASS_MAP = {1: "Defect_1", 2: "Defect_2", 3: "Defect_3", 4: "Defect_4"}

# ─────────────────────────────────────────────────────────────
# Transforms & Model (from your script)
# ─────────────────────────────────────────────────────────────
def get_inference_transform():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((800, 800), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_model():
    backbone = resnet_fpn_backbone(
        "resnet50", weights=ResNet50_Weights.IMAGENET1K_V1, trainable_layers=0
    )
    return FasterRCNN(backbone, num_classes=NUM_CLASSES)

@st.cache_resource  # Cache model loading for performance
def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device

# ─────────────────────────────────────────────────────────────
# Inference Function (adapted for single image)
# ─────────────────────────────────────────────────────────────
def run_inference(img: Image.Image, model, device, transform, score_threshold=SCORE_THRESHOLD) -> List[Dict]:
    img_tensor, _ = transform(img, {})
    img_tensor = img_tensor.to(device).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        output = model(img_tensor)[0]

    keep = output["scores"] >= score_threshold

    boxes = output["boxes"][keep].cpu().tolist()
    labels = output["labels"][keep].cpu().tolist()
    scores = output["scores"][keep].cpu().tolist()

    detections = []
    for box, lbl, sc in zip(boxes, labels, scores):
        class_name = CLASS_MAP.get(lbl, "Unknown")
        detections.append({
            "class": class_name,
            "score": round(sc, 3),
            "box": [round(v, 1) for v in box]
        })
    
    return detections

# ─────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────
st.title("Steel Defect Detection App")
st.write("Upload multiple images from your folder (JPG/PNG/JPEG). The app will run defect detection on each and display results in a scrollable table.")

# Load model once
model, device = load_model(WEIGHTS_PATH)
transform = get_inference_transform()

# File uploader for multiple images (simulates folder by selecting all files)
uploaded_files = st.file_uploader("Choose images from folder...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for idx, file in enumerate(uploaded_files):
        try:
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            detections = run_inference(img, model, device, transform)
            
            results.append({
                "Filename": file.name,
                "Detections": detections if detections else "No defects detected",
                "Num Detections": len(detections)
            })
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((idx + 1) / total_files)
    
    if results:
        # Display as scrollable DataFrame
        df = pd.DataFrame(results)
        st.dataframe(df, height=400, use_container_width=True)  # Scrollable table (adjust height as needed)
    else:
        st.warning("No valid images processed.")
