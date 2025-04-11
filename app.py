import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
from model import ConvAutoencoder
from matplotlib import pyplot as plt
from tempfile import NamedTemporaryFile
from PIL import Image

st.set_page_config(layout="wide", page_title="AI-Powered Anomaly Detection")
st.markdown("# AI-Powered Anomaly Detection in Surveillance")

# Sidebar: model info, threshold, download, credits
st.sidebar.title("🔧 Settings & Info")
thresh_slider = st.sidebar.slider("Anomaly Threshold", min_value=0.0, max_value=0.1, value=0.005, step=0.0005)
st.sidebar.markdown("""---
### ℹ️ Model Info
- Type: CNN Autoencoder
- Input: Grayscale 128x128
- Encoder: 3 Conv layers
- Decoder: 3 Transposed Conv layers

### 👩‍💻 Credits
Developed by Diya Manoj
""")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
checkpoint = torch.load("cnn_autoencoder_anomaly.pt", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (128, 128))
    tensor = transform(frame).unsqueeze(0).to(device)
    return tensor, frame

def get_reconstruction_error(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2).item()

# Upload input
uploaded_file = st.file_uploader("Upload video or image", type=["jpg", "jpeg", "png", "mp4", "avi", "mpeg4"])

if uploaded_file:
    filename = uploaded_file.name
    file_ext = filename.split(".")[-1].lower()
    bytes_data = uploaded_file.read()

    with NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(bytes_data)
        temp_path = tmp.name

    # Process image
    if file_ext in ["jpg", "jpeg", "png"]:
        image = Image.open(temp_path).convert("L").resize((128, 128))
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed = model(input_tensor)
        error = get_reconstruction_error(input_tensor, reconstructed)

        st.image(image, caption=f"Reconstruction Error: {error:.4f}", use_column_width=True)

    # Process video
    elif file_ext in ["mp4", "avi", "mpeg4"]:
        cap = cv2.VideoCapture(temp_path)
        errors = []
        frames = []
        heatmaps = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            tensor, raw_frame = preprocess_frame(frame)
            with torch.no_grad():
                reconstructed = model(tensor)

            error = get_reconstruction_error(tensor, reconstructed)
            errors.append(error)
            frames.append(raw_frame)

        cap.release()

        # Show graph
        st.line_chart(errors)

        # Show top anomalies
        anomalies = [(i, e) for i, e in enumerate(errors) if e > thresh_slider]
        anomalies.sort(key=lambda x: x[1], reverse=True)

        st.markdown("### 🔥 Top Anomalies")
        col1, col2, col3 = st.columns(3)
        if anomalies:
            for i, (idx, err) in enumerate(anomalies[:3]):
                with [col1, col2, col3][i % 3]:
                    st.image(frames[idx], caption=f"Frame {idx} - Error: {err:.4f}", use_column_width=True)

        # Save anomaly logs
        log_txt = "Frame,Error\n" + "\n".join([f"{idx},{err:.6f}" for idx, err in anomalies])
        with open("anomaly_log.csv", "w") as f:
            f.write(log_txt)

        st.sidebar.download_button("📥 Download Anomaly Log", data=log_txt, file_name="anomaly_log.csv")
