import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from physics_engine import calculate_fractal_dimension, generate_entropy_heatmap
from database import init_db, save_result, get_unsynced_count
from aws_sync import sync_db_to_aws
from model import FractalHybridCNN

# 1. UI Polish - Set Wide Layout & Custom Theme
st.set_page_config(page_title="FractalLens Edge Dashboard", page_icon="💠", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0e1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .main-header { font-size: 2.2rem; color: #4facfe; font-weight: bold; margin-bottom: 0;}
    .sub-header { color: #888; margin-top: 0; font-size: 1.1rem;}
    </style>
""", unsafe_allow_html=True)

# 2. Load the REAL PyTorch Model
@st.cache_resource
def load_model():
    model = FractalHybridCNN(num_classes=2)
    # Load the weights you just trained!
    model.load_state_dict(torch.load('fractallens_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

ai_model = load_model()
init_db()

# 3. Header & AWS Sync
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header">💠 FractalLens: Edge AI Diagnostics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Physics-Informed Anomaly Detection for Rural Healthcare</p>', unsafe_allow_html=True)
with col2:
    unsynced = get_unsynced_count()
    st.info(f"📶 Offline Edge DB Active\n\nPending AWS Sync: **{unsynced}** records")
    if st.button("🔄 Sync to AWS S3", use_container_width=True):
        with st.spinner("Encrypting and Syncing to AWS..."):
            success, message = sync_db_to_aws()
            if success: st.success(message)
            else: st.error(message)

st.divider()

# 4. Deep Learning Image Preprocessor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Main Dashboard Layout
col_input, col_results = st.columns([1, 1.5], gap="large")

with col_input:
    st.subheader("📋 Patient Input")
    patient_name = st.text_input("Patient ID / Name", placeholder="e.g., Patient_104_Rural")
    uploaded_file = st.file_uploader("Upload Low-Res Scan (X-Ray/CT)", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption="Raw Uploaded Scan", use_container_width=True)

with col_results:
    st.subheader("🔬 AI Diagnostic Engine")
    if uploaded_file:
        if st.button("Run Hybrid PINN Inference", type="primary", use_container_width=True):
            with st.spinner("Extracting Minkowski-Bouligand Dimensions & Running Inference..."):
                
                # A. Physics Engine (Math)
                img_cv2 = np.array(image_pil.convert('L'))
                global_fd = calculate_fractal_dimension(img_cv2)
                heatmap_overlay, _ = generate_entropy_heatmap(img_cv2, grid_size=16)
                
                color_img = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR)
                blended = cv2.addWeighted(color_img, 0.6, heatmap_overlay, 0.4, 0)
                
                # B. Real PyTorch AI Inference
                img_tensor = transform(image_pil).unsqueeze(0)
                fd_tensor = torch.tensor([global_fd], dtype=torch.float32)
                
                with torch.no_grad():
                    outputs = ai_model(img_tensor, fd_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
                # Assuming Class 1 is Anomaly (Pneumonia) based on Kaggle dataset structure
                anomaly_prob = probabilities[1].item() * 100
                is_anomaly = anomaly_prob > 50.0
                prediction = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
                
                # C. Display Professional Results
                st.image(blended, caption="Entropy Heatmap (Red = Pathological Chaos/Roughness)", channels="BGR", use_container_width=True)
                
                st.markdown("### Analysis Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Fractal Dimension ($D_f$)", f"{global_fd:.3f}")
                m2.metric("Anomaly Probability", f"{anomaly_prob:.1f}%")
                m3.metric("Final Status", prediction, delta_color="inverse" if is_anomaly else "normal")
                
                # D. Save to DB
                if patient_name:
                    save_result(patient_name, global_fd, prediction)
                    st.success("✅ Securely saved to Edge SQLite. Ready for opportunistic AWS sync.")
                else:
                    st.warning("⚠️ Record not saved. Please enter a Patient ID.")