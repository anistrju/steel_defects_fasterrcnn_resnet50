import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import io
import numpy as np
import pandas as pd
from typing import List, Tuple

# ========================== MODEL DEFINITION ==========================
# Paste this from your notebook (or use this minimal working version)
class UNetResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # 64, 128x800
        self.enc2 = resnet.layer1                                          # 64, 128x800
        self.enc3 = resnet.layer2                                          # 128, 64x400
        self.enc4 = resnet.layer3                                          # 256, 32x200
        self.enc5 = resnet.layer4                                          # 512, 16x100

        self.up5 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, stride=2), nn.ReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose2d(256+256, 128, 2, stride=2), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128+128, 64, 2, stride=2), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64+64, 64, 2, stride=2), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64+64, 32, 2, stride=2), nn.ReLU())

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.up5(e5)
        d4 = self.up4(torch.cat([d5, e4], dim=1))
        d3 = self.up3(torch.cat([d4, e3], dim=1))
        d2 = self.up2(torch.cat([d3, e2], dim=1))
        d1 = self.up1(torch.cat([d2, e1], dim=1))

        return self.final(d1)   # (B, 4, 256, 1600)

# ========================== CONFIG ==========================
NUM_CLASSES = 4
THRESHOLD_DEFAULT = 0.5
MODEL_PATH = "best_model.pth"          # ← change if needed
HEIGHT, WIDTH = 256, 1600

CLASS_MAP = {1: "Defect_1", 2: "Defect_2", 3: "Defect_3", 4: "Defect_4"}
COLOR_MAP = {
    1: (255, 80, 80, 180),
    2: (80, 200, 80, 180),
    3: (80, 80, 255, 180),
    4: (255, 220, 60, 180),
}

# ========================== TRANSFORMS ==========================
inference_transform = T.Compose([
    T.ToImage(),
    T.Resize((HEIGHT, WIDTH), antialias=True),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========================== MODEL LOAD ==========================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetResNet18(num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device

model, device = load_model()

# ========================== VISUALIZATION ==========================
def create_visualizations(pil_img: Image.Image, mask_tensor: torch.Tensor, thresh: float) -> Tuple[Image.Image, Image.Image]:
    original = pil_img.copy()
    img_rgba = pil_img.convert("RGBA")
    gray = ImageOps.grayscale(img_rgba).convert("RGBA")
    darkened = ImageEnhance.Brightness(gray).enhance(0.45)

    draw = ImageDraw.Draw(darkened, "RGBA")
    mask_np = (mask_tensor.cpu().numpy() > thresh).astype(np.uint8)  # (4, H, W)

    for c in range(NUM_CLASSES):
        if mask_np[c].max() == 0:
            continue
        color = COLOR_MAP.get(c+1, (255, 255, 255, 180))
        mask_img = Image.fromarray(mask_np[c] * 255).convert("L")
        overlay = Image.new("RGBA", pil_img.size, color)
        darkened.paste(overlay, (0, 0), mask_img)

    highlighted = darkened.convert("RGB")
    return original, highlighted

# ========================== INFERENCE ==========================
def run_inference(pil_img: Image.Image, model, device, transform, thresh: float):
    tensor = transform(pil_img).unsqueeze(0).to(device)          # (1, 3, 256, 1600)

    with torch.no_grad():
        logits = model(tensor)                                   # (1, 4, 256, 1600)
        probs = torch.sigmoid(logits)[0]                         # (4, 256, 1600)

    # Per-class binary masks
    binary_masks = (probs > thresh).float()

    # Classes present
    present_classes = [CLASS_MAP[i+1] for i in range(NUM_CLASSES) if binary_masks[i].max() > 0]

    return binary_masks, present_classes

# ========================== STREAMLIT APP ==========================
st.title("Steel Defect Segmentation (U-Net)")
st.markdown("Upload steel strip images → pixel-level defect masks → colored overlay")

model, device = load_model()

thresh = st.slider("Mask threshold", 0.1, 0.9, THRESHOLD_DEFAULT, 0.05)

uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    progress = st.progress(0)

    for i, file in enumerate(uploaded_files):
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        mask_tensor, present_classes = run_inference(img, model, device, inference_transform, thresh)
        orig, high = create_visualizations(img, mask_tensor, thresh)

        # Thumbnails
        orig_small = orig.resize((300, 48))   # keep aspect ~1600:256
        high_small = high.resize((300, 48))

        buf_o = io.BytesIO(); orig_small.save(buf_o, "PNG")
        buf_h = io.BytesIO(); high_small.save(buf_h, "PNG")

        results.append({
            "Filename": file.name,
            "Original": buf_o.getvalue(),
            "Highlighted": buf_h.getvalue(),
            "Defects Detected": ", ".join(present_classes) if present_classes else "None",
            "Num Classes": len(present_classes),
            "Mask Tensor Shape": str(mask_tensor.shape)
        })

        progress.progress((i+1) / len(uploaded_files))

    # Display table
    df = pd.DataFrame(results)

    def img_formatter(b):
        import base64
        return f'<img src="data:image/png;base64,{base64.b64encode(b).decode()}" width="300"/>'

    st.subheader(f"Results ({len(results)} images)")
    st.markdown(
        df.style
        .format({"Original": img_formatter, "Highlighted": img_formatter})
        .to_html(escape=False),
        unsafe_allow_html=True
    )

    with st.expander("Raw mask info"):
        st.json([{r["Filename"]: {"classes": r["Defects Detected"], "shape": r["Mask Tensor Shape"]}} for r in results])