import streamlit as st
import torch
import cv2
import numpy as np
import datetime
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Vehicle Damage Intelligence",
    layout="wide"
)

# -------------------------------------------------
# PREMIUM UI STYLING
# -------------------------------------------------

st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #23272F 0%, #2C313C 100%);
    box-shadow: 0px 4px 25px rgba(0, 0, 0, 0.5);
    border: 1px solid #2E3440;
    margin-bottom: 15px;
}

.metric-title {
    font-size: 14px;
    color: #B0B8C5;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 26px;
    font-weight: 600;
    color: #FFFFFF;
}

.severity-card {
    background: linear-gradient(135deg, #1F2933 0%, #323F4B 100%);
    border-radius: 15px;
    padding: 20px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚗 Vehicle Damage Intelligence Platform")
st.markdown("AI-powered inspection & severity analytics system")
st.markdown("---")

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

@st.cache_resource
def load_model():
    model = YOLO("runs/classify/train/weights/best.pt")
    net = model.model

    class WrappedClassificationModel(torch.nn.Module):
        def __init__(self, model_core):
            super().__init__()
            self.model_core = model_core

        def forward(self, x):
            output = self.model_core(x)
            if isinstance(output, tuple):
                return output[0]
            return output

    wrapped_net = WrappedClassificationModel(net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped_net.to(device)
    wrapped_net.eval()

    target_layer = net.model[9].conv
    cam = GradCAM(model=wrapped_net, target_layers=[target_layer])

    return wrapped_net, cam, device

wrapped_net, cam, device = load_model()

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    inspection_id = "INS-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_resized = cv2.resize(img, (224, 224))
    rgb_img = img_resized[:, :, ::-1] / 255.0

    input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    input_tensor.requires_grad = True

    # ----------------------------
    # PREDICTION
    # ----------------------------

    output = wrapped_net(input_tensor)
    probs_tensor = torch.softmax(output, dim=1)
    probs = probs_tensor.detach().cpu().numpy()[0]

    labels = ["Damage", "No Damage"]
    pred_class = torch.argmax(probs_tensor, dim=1).item()
    confidence = probs[pred_class]

    # ----------------------------
    # GRADCAM
    # ----------------------------

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # ----------------------------
    # BOUNDING BOX
    # ----------------------------

    heatmap_resized = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (
        heatmap_resized.max() - heatmap_resized.min() + 1e-8
    )

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    _, thresh = cv2.threshold(heatmap_uint8, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bbox_img = img.copy()
    damage_category = "Unknown"

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        cv2.rectangle(bbox_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)

        if y > img.shape[0] * 0.5:
            damage_category = "Functional Damage"
        else:
            damage_category = "Cosmetic Damage"

    # ----------------------------
    # SEVERITY CALCULATION
    # ----------------------------

    h, w = heatmap_resized.shape
    damage_mask = heatmap_resized > 0.35
    damage_area_ratio = np.sum(damage_mask) / (h * w)
    critical_mask = damage_mask[int(h * 0.5):h, :]
    critical_ratio = np.sum(critical_mask) / (h * w)

    if critical_ratio > 0.02:
        severity = "HIGH"
        severity_color = "#E74C3C"
        severity_progress = 90
    elif damage_area_ratio > 0.08:
        severity = "MEDIUM"
        severity_color = "#F39C12"
        severity_progress = 60
    else:
        severity = "LOW"
        severity_color = "#2ECC71"
        severity_progress = 30

    # -------------------------------------------------
    # DASHBOARD LAYOUT
    # -------------------------------------------------

    st.markdown(f"**Inspection ID:** `{inspection_id}`")
    st.markdown("---")

    left, right = st.columns([1.1, 1])

    # LEFT PANEL
    with left:
        st.markdown("### 🔍 Damage Localization")
        st.image(bbox_img[:, :, ::-1], use_container_width=True)

        st.markdown("### 🧠 AI Attention Map")
        st.image(visualization, use_container_width=True)

    # RIGHT PANEL
    with right:

        colA, colB = st.columns(2)

        with colA:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="metric-title">Prediction</div>
                <div class="metric-value">{labels[pred_class]}</div>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="metric-title">Confidence</div>
                <div class="metric-value">{confidence:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### ⚠ Severity Analysis")
        st.progress(severity_progress)

        st.markdown(f"""
        <div class="severity-card" style="border-left:6px solid {severity_color};">
            <div class="metric-title">Severity Level</div>
            <div class="metric-value" style="color:{severity_color};">{severity}</div>
            <br>
            <div class="metric-title">Damage Category</div>
            <div class="metric-value">{damage_category}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 System Recommendation")

        if severity == "HIGH":
            st.error("⚠ Immediate physical inspection required.")
        elif severity == "MEDIUM":
            st.warning("⚠ Service recommended within short interval.")
        else:
            st.success("✔ Minor cosmetic issue detected.")

        report = {
            "Inspection_ID": inspection_id,
            "Prediction": labels[pred_class],
            "Confidence": round(float(confidence), 3),
            "Severity": severity,
            "Damage_Category": damage_category,
        }

        st.download_button(
            label="📄 Download Inspection Report",
            data=str(report),
            file_name=f"{inspection_id}_report.json",
            mime="application/json"
        )
