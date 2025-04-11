import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

# ------------------------
# Define the Model
# ------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> (16, 64, 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> (64, 16, 16)
            nn.ReLU()
        )

        # Decoder
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
# ------------------------
# Load the Model
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)

try:
    model.load_state_dict(torch.load("cnn_autoencoder_anomaly.pt", map_location=device))
    model.eval()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model loading failed: {e}")

# ------------------------
# Streamlit UI
# ------------------------
st.title("🔍 AI-Powered Anomaly Detection in Surveillance")

uploaded_file = st.file_uploader("Upload an image for anomaly detection", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((128, 128))
    img_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # Calculate reconstruction error
    loss_fn = nn.MSELoss()
    reconstruction_error = loss_fn(output, img_tensor).item()
    st.metric(label="Reconstruction Error", value=f"{reconstruction_error:.6f}")

    # Show images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        reconstructed = output.squeeze().cpu().numpy()
        st.image(reconstructed, caption="Reconstructed Image", use_column_width=True)

    # Show heatmap
    error_map = np.abs(img_tensor.squeeze().cpu().numpy() - reconstructed)
    heatmap = cv2.applyColorMap(np.uint8(255 * error_map), cv2.COLORMAP_JET)
    st.image(heatmap, caption="🔍 Anomaly Heatmap", use_column_width=True)
