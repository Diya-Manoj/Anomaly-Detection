import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# Define the CNN Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> (16, 64, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> (64, 16, 16)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # -> (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> (16, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # -> (1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)

try:
    model.load_state_dict(torch.load("cnn_autoencoder_anomaly.pt", map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Model loading failed: {e}")

# Helper function to process a single frame
def process_frame(frame, model, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        loss = F.mse_loss(output, tensor)

    recon = output.squeeze().cpu().numpy()
    return resized, recon, loss.item()

# Streamlit UI
st.title("AI-Powered Anomaly Detection in Surveillance")
uploaded_file = st.file_uploader("Upload video or image", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file:
    if uploaded_file.type.startswith("video"):
        tfile = open("temp_video.mp4", 'wb')
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")

        stframe = st.empty()
        error_plot = st.line_chart([], height=200, use_container_width=True)
        errors = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized, recon, error = process_frame(frame, model, device)
            errors.append(error)

            recon_display = (recon * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(cv2.convertScaleAbs(cv2.absdiff(resized, recon_display)), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)

            stframe.image(overlay, channels="BGR", caption=f"Reconstruction Error: {error:.4f}")
            error_plot.add_rows([error])

        cap.release()

    else:
        img = Image.open(uploaded_file).convert('L')
        img = img.resize((128, 128))
        tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            loss = F.mse_loss(output, tensor)
            recon = output.squeeze().cpu().numpy()

        heatmap = cv2.applyColorMap(cv2.convertScaleAbs(cv2.absdiff(np.array(img), (recon * 255).astype(np.uint8))), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(img.convert('RGB')), 0.6, heatmap, 0.4, 0)

        st.image(img, caption="Original Image")
        st.image(overlay, caption=f"Reconstruction Heatmap (Error: {loss.item():.4f})")
