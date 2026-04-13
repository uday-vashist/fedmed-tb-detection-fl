# app/streamlit_app.py
# Run with: streamlit run app/streamlit_app.py

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.data_loader import get_transforms
from src.config import MODELS_DIR, DEVICE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedMed – TB Detection",
    page_icon="🫁",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .title  { text-align: center; font-size: 2.2rem; font-weight: 700; color: #4B4ACF; }
        .sub    { text-align: center; color: #666; margin-bottom: 1.5rem; }
        .result-box {
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-top: 1rem;
            font-size: 1.1rem;
        }
        .tb      { background: #fdecea; border-left: 5px solid #e53935; color: #b71c1c; }
        .normal  { background: #e8f5e9; border-left: 5px solid #43a047; color: #1b5e20; }
        .footer  { text-align: center; color: #aaa; font-size: 0.8rem; margin-top: 3rem; }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🫁 FedMed: TB Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Privacy-Preserving Tuberculosis Screening via Federated Learning</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Selection")
    model_choice = st.radio(
        "Select model:",
        ["Baseline (Centralized)", "Federated Learning"],
        help="Baseline: trained on full data centrally. FL: trained across 3 simulated hospitals using FedAvg."
    )
    model_file = "baseline_best.pth" if model_choice == "Baseline (Centralized)" else "fl_best.pth"

    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
**FedMed** uses a ResNet-50 model trained on a combined dataset of 3396 chest X-rays
(TBX11K + Shenzhen + TB Chest) using Federated Learning with the Flower framework.

**How to use:**
1. Select a model above
2. Upload a chest X-ray (PNG / JPG)
3. Click **Analyze X-Ray**
4. View prediction and confidence

**Models:**
- **Baseline**: Centralized training, 95.88% test accuracy
- **FL**: Federated across 3 hospitals, 87.03% val accuracy

**Disclaimer:** Research purposes only. Not a substitute for clinical diagnosis.
    """)
    st.divider()
    st.caption("Model: ResNet-50 | Framework: PyTorch + Flower FL")

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_file):
    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        return None
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model(model_file)

if model is None:
    st.warning(
        f"⚠️ Model not found at `results/models/{model_file}`. "
        "Run training first.",
        icon="⚠️",
    )
    st.stop()

# ── Show which model is active ────────────────────────────────────────────────
st.info(f"**Active model:** {model_choice} (`{model_file}`)", icon="🤖")

transform = get_transforms(train=False)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    st.divider()

    if st.button("🔬 Analyze X-Ray", type="primary", use_container_width=True):
        with st.spinner("Analyzing image…"):
            try:
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs      = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)

                label     = "TB Detected" if predicted.item() == 1 else "Normal"
                conf_pct  = confidence.item() * 100
                css_class = "tb" if predicted.item() == 1 else "normal"
                icon      = "🔴" if predicted.item() == 1 else "🟢"

                st.markdown("### 📊 Results")
                st.markdown(
                    f'<div class="result-box {css_class}">'
                    f'<strong>{icon} Prediction:</strong> {label}<br>'
                    f'<strong>Confidence:</strong> {conf_pct:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### Class Probabilities")
                probs     = probabilities[0].cpu().tolist()
                prob_data = {"Normal": probs[0], "TB": probs[1]}
                st.bar_chart(prob_data)

                if predicted.item() == 1:
                    st.error(
                        "⚠️ TB indicators detected. Please consult a qualified physician for confirmation.",
                        icon="🏥",
                    )
                else:
                    st.success(
                        "✅ No TB indicators detected. Regular screening is still recommended.",
                        icon="✅",
                    )

            except Exception as e:
                st.error(f"Error during analysis: {e}")
else:
    st.info("👆 Please upload a chest X-ray image to get started.", icon="📁")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="footer">FedMed — Privacy-Preserving TB Detection | Research Project 2025</p>',
    unsafe_allow_html=True
)
