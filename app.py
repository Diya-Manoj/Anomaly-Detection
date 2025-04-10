import streamlit as st
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from model import ConvAutoencoder  # Make sure your model class is in model.py or adjust import
from utils import extract_frames, FrameDataset, overlay_heatmap  # Assume you modularized these

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# App title
st.title("🔍 AI-Powered Anomaly Detection in Surveillance Video")

# Upload video
uploaded_video = st.file_uploader("Upload a CCTV video file", type=["mp4", "avi"])
if uploaded_video:
    # Save uploaded file
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.video(video_path)
    
    st.info("Extracting frames and processing...")
    extract_frames(video_path, "frames")

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = FrameDataset("frames", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Load model
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load("cnn_autoencoder_anomaly.pt", map_location=device))  # Load your trained model
    model.eval()

    # Inference
    reconstruction_errors = []
    original_frames = []
    reconstructed_frames = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)

            # Save for heatmap
            original_frames.extend(batch.cpu())
            reconstructed_frames.extend(output.cpu())

            loss = torch.mean((batch - output) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(loss.cpu().numpy())

    # Calculate threshold
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)

    # Display errors
    st.subheader("📉 Reconstruction Error Distribution")
    st.line_chart(reconstruction_errors)

    # Heatmap Overlay
    st.subheader("🔥 Detected Anomalies with Heatmap Overlay")
    for i, (orig, recon, err) in enumerate(zip(original_frames, reconstructed_frames, reconstruction_errors)):
        if err > threshold:
            heatmap_overlay = overlay_heatmap(orig, recon)
            st.image(heatmap_overlay, caption=f"Frame {i} - Anomaly Score: {err:.4f}", use_column_width=True)

    st.success("✅ Detection complete.")
