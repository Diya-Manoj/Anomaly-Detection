import streamlit as st
import numpy as np
import cv2
import os
from model import ConvAutoencoder
from matplotlib import pyplot as plt
from tempfile import NamedTemporaryFile
from PIL import Image
import torch
import torch.nn as nn  # Added import for nn.Module usage
import torchvision.transforms as transforms

# Load model
@st.cache_resource
def load_model(model_path='model.pth'):
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Title and Sidebar
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.sidebar.title("🔍 Anomaly Detector")
st.sidebar.markdown("Adjust settings or upload video.")

threshold = st.sidebar.slider("Anomaly Threshold (0.01 is frequently used", 0.0, 1.0, 0.1, step=0.01)
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Model Info**\n- Type: CNN Autoencoder\n- Framework: PyTorch")

# Upload Video
video_file = st.file_uploader("Upload a surveillance video", type=["mp4", "avi", "mov"])
top_anomalies = []

if video_file:
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stframe = st.empty()

    anomaly_scores = []
    frame_list = []

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    with st.spinner("🔍 Analyzing video..."):
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame_gray, (128, 128))
            frame_tensor = transform(frame_resized).unsqueeze(0)

            with torch.no_grad():
                reconstructed = model(frame_tensor)
                loss = torch.mean((frame_tensor - reconstructed) ** 2).item()
                anomaly_scores.append(loss)
                frame_list.append(frame)

            if loss > threshold:
                top_anomalies.append((loss, frame))

    cap.release()

    st.success("✅ Analysis Complete!")

    # Plot error graph
    st.subheader("📈 Anomaly Score Over Time")
    fig, ax = plt.subplots()
    ax.plot(anomaly_scores, label="Reconstruction Error")
    ax.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Error")
    ax.legend()
    st.pyplot(fig)

    # Top anomalies
    if top_anomalies:
        top_anomalies.sort(reverse=True, key=lambda x: x[0])
        st.subheader("🔥 Top Anomalies")
        cols = st.columns(3)

        for i, (score, frame) in enumerate(top_anomalies[:6]):
            with cols[i % 3]:
                st.image(frame, caption=f"Score: {score:.4f}", use_container_width=True)
    else:
        st.info("No significant anomalies detected above the threshold.")

else:
    st.warning("📤 Please upload a video file to begin anomaly detection.")
