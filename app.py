import base64
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import io
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import streamlit.components.v1 as components


# ========================== MODEL DEFINITION ==========================
# Paste this from your notebook (or use this minimal working version)
class UNetResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=False, decoder_mode="add", dropout=0.0):
        super().__init__()
        # Encoder backbone
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        del base.fc, base.avgpool

        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.enc2 = nn.Sequential(base.maxpool, base.layer1)
        self.enc3 = base.layer2
        self.enc4 = base.layer3
        self.enc5 = base.layer4

        self.mode = decoder_mode

        def up_block(in_ch, out_ch, use_concat=False):
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            if use_concat:
                layers += [
                    nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ]
            return nn.Sequential(*layers)

        if self.mode == "add":
            self.up4 = up_block(512, 256)
            self.up3 = up_block(256, 128)
            self.up2 = up_block(128, 64)
            self.up1 = up_block(64, 64)
        else:  # concat
            self.up4 = up_block(512, 256, use_concat=True)
            self.up3 = up_block(256, 128, use_concat=True)
            self.up2 = up_block(128, 64, use_concat=True)
            self.up1 = up_block(64, 64, use_concat=True)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        if self.mode == "add":
            d4 = self.up4(e5) + e4
            d3 = self.up3(d4) + e3
            d2 = self.up2(d3) + e2
            d1 = self.up1(d2) + e1
        else:  # concat
            d4 = self.up4(torch.cat([F.interpolate(e5, size=e4.shape[2:], mode="bilinear", align_corners=False), e4], 1))
            d3 = self.up3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], 1))
            d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], 1))
            d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], 1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out

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
    
    # Create model with EXACT same parameters as training
    model = UNetResNet18(
        num_classes=4,                  # matches CFG["NUM_CLASSES"]
        pretrained=False,               # we load custom weights → don't load ImageNet again
        decoder_mode="add",             # critical: matches CFG["DECODER_MODE"]
        dropout=0.0                     # default from your class
    )
    
    # Load checkpoint
    checkpoint_path = "best_model.pth"  # adjust if your file has different name
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle possible DataParallel / DistributedDataParallel wrapper
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load weights — strict=True will tell us immediately if there's still mismatch
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()
    
    st.success(f"Model loaded successfully on {device}")
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

        # Thumbnails — small size so table doesn't explode
        orig_small = orig.resize((320, 51))   # ≈1600:256 aspect ratio
        high_small = high.resize((320, 51))

        buf_o = io.BytesIO()
        orig_small.save(buf_o, "PNG")
        buf_o.seek(0)

        buf_h = io.BytesIO()
        high_small.save(buf_h, "PNG")
        buf_h.seek(0)

        results.append({
            "Filename": file.name,
            "Original": buf_o.getvalue(),
            "Highlighted": buf_h.getvalue(),
            "Defects Detected": ", ".join(present_classes) if present_classes else "None",
            "Num Classes": len(present_classes)
        })

        progress.progress((i + 1) / len(uploaded_files))

    # Create DataFrame
    df = pd.DataFrame(results)

    # ────────────────────────────────────────────────
    #   IMAGE FORMATTER — small thumbnails
    # ────────────────────────────────────────────────
    def thumbnail_formatter(b):
        import base64
        b64 = base64.b64encode(b).decode()
        return f"""
        <img 
            src="data:image/png;base64,{b64}" 
            width="240" 
            style="border-radius:4px; object-fit:cover;"
        />
        """

    # ────────────────────────────────────────────────
    #   CSS: hover zoom effect + better table behavior
    # ────────────────────────────────────────────────
    st.markdown("""
    <style>
        /* Make table scrollable horizontally if needed */
        div[data-testid="stTable"] {
            overflow-x: auto;
            max-width: 100%;
        }

        /* Table cell styling */
        .stTable td, .stTable th {
            padding: 8px 10px !important;
            text-align: center;
            vertical-align: middle;
            white-space: nowrap;
        }

        /* Image styling */
        .stTable img {
            transition: transform 0.25s ease;
            transform-origin: top left;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            border-radius: 4px;
        }

        /* Zoom on hover */
        .stTable img:hover {
            transform: scale(2.4);          /* adjust this value: 2.0–3.5 */
            z-index: 10;
            box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        }

        /* Make sure cells don't collapse */
        .stTable td[data-testid="cell"] {
            min-width: 260px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────
    #   Display results table
    # ────────────────────────────────────────────────
    st.subheader(f"Results ({len(results)} images)")

    st.markdown(
        df.style
        .format({
            "Original": thumbnail_formatter,
            "Highlighted": thumbnail_formatter
        })
        .set_properties(
            **{
                "text-align": "center",
                "min-width": "260px"     # helps prevent squeezing
            },
            subset=["Original", "Highlighted"]
        )
        .to_html(escape=False),
        unsafe_allow_html=True
    )

    # Optional: keep your raw info expander
    with st.expander("Raw mask info"):
        st.json([{r["Filename"]: {"classes": r["Defects Detected"]}} for r in results])