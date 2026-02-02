import streamlit as st
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2 as T
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import io
import pandas as pd
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────────────────────
# Configuration (from your script)
# ─────────────────────────────────────────────────────────────
NUM_CLASSES = 5
SCORE_THRESHOLD = 0.3
WEIGHTS_PATH = "/app/fasterrcnn_severstal.pth"  # Adjust if mounted elsewhere

CLASS_MAP = {1: "Defect_1", 2: "Defect_2", 3: "Defect_3", 4: "Defect_4"}

COLOR_MAP = {
    1: (255, 80, 80, 180),   # reddish
    2: (80, 200, 80, 180),   # greenish
    3: (80, 80, 255, 180),   # blueish
    4: (255, 220, 60, 180),  # yellow
}
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
# Visualization: create original + highlighted version
# ─────────────────────────────────────────────────────────────
def create_visualizations(pil_img: Image.Image, detections: List[Dict], thresh: float = 0.3) -> Tuple[Image.Image, Image.Image]:
    # Original
    original = pil_img.copy()

    # Highlighted version: grayscale + darkened + colored boxes
    gray = ImageOps.grayscale(pil_img)
    enhancer = ImageEnhance.Brightness(gray)
    darkened = enhancer.enhance(0.45)  # 0.3–0.6 range usually looks good

    draw = ImageDraw.Draw(darkened, "RGBA")

    kept_dets = [d for d in detections if d["score"] >= thresh]

    for det in kept_dets:
        x1, y1, x2, y2 = det["box"]
        label = det["class"]
        score = det["score"]
        color = COLOR_MAP.get(int(label.split("_")[1]), (220, 220, 60, 200))

        # Semi-transparent fill + solid outline
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color[:3] + (255,),
            fill=color,
            width=4
        )

        # Text label
        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font_size=14)
        text_w = bbox[2] - bbox[0]
        draw.rectangle(
            (x1, y1 - 22, x1 + text_w + 8, y1),
            fill=(0, 0, 0, 180)
        )
        draw.text((x1 + 4, y1 - 20), text, fill=(255, 255, 255), font_size=14)

    highlighted = darkened.convert("RGB")
    return original, highlighted
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
        class_name = CLASS_MAP.get(lbl, f"Class_{lbl}")
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

score_thresh = st.slider(
    "Confidence threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(SCORE_THRESHOLD),
    step=0.05,
    help="Lower = more detections (may include false positives)"
)

# File uploader for multiple images (simulates folder by selecting all files)
uploaded_files = st.file_uploader("Choose images from folder...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    progress = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)}) ...")

        try:
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            detections = run_inference(img, model, device, transform, score_thresh)

            orig_img, high_img = create_visualizations(img, detections, score_thresh)

            # Resize for table preview (small thumbnails)
            orig_small = orig_img.copy().resize((180, 180))
            high_small = high_img.copy().resize((180, 180))

            buf_orig = io.BytesIO()
            buf_high = io.BytesIO()
            orig_small.save(buf_orig, format="PNG")
            high_small.save(buf_high, format="PNG")

            results.append({
                "Filename": file.name,
                "Original": buf_orig.getvalue(),
                "Highlighted": buf_high.getvalue(),
                "Num Detections": len(detections),
                "Detections": detections if detections else "None"
            })

        except Exception as e:
            st.error(f"Error on {file.name}: {str(e)}")

        progress.progress((i + 1) / len(uploaded_files))

    status_text.text("Processing complete.")

    if results:
        df = pd.DataFrame(results)

        def image_formatter(img_bytes):
            import base64
            b64 = base64.b64encode(img_bytes).decode()
            return f'<img src="data:image/png;base64,{b64}" width="180" />'

        st.subheader(f"Results ({len(results)} images)")

        st.markdown(
            df.style
            .format({
                "Original": image_formatter,
                "Highlighted": image_formatter
            })
            .set_properties(**{'text-align': 'center'}, subset=["Original", "Highlighted"])
            .to_html(escape=False),
            unsafe_allow_html=True
        )

        # Optional: expandable raw data
        with st.expander("Raw detection details (JSON-like)", expanded=False):
            st.json([{"filename": r["Filename"], "detections": r["Detections"]} for r in results])

    else:
        st.info("No images were successfully processed.")