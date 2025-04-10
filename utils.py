import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch

def extract_frames(video_path, output_dir, resize=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output_dir}/frame_{idx:04d}.jpg", gray)
        idx += 1
    cap.release()

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames = sorted(os.listdir(frames_dir))
        self.dir = frames_dir
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.frames[idx])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

def overlay_heatmap(orig_tensor, recon_tensor):
    orig = orig_tensor.squeeze().numpy()
    recon = recon_tensor.squeeze().numpy()
    error_map = np.abs(orig - recon)

    # Normalize
    error_map -= error_map.min()
    error_map /= error_map.max()

    # Convert to heatmap
    heatmap = cv2.applyColorMap((error_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    orig_img = (orig * 255).astype(np.uint8)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

    blended = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    return blended
