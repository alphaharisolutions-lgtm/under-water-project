# 🌊 DeepSea Restoration AI: Hybrid Underwater Image Enhancement

A premium, high-fidelity underwater image restoration system combining **Physics-based Variational Imaging Models** with **Deep Learning Residual U-Net** architectures.

---

## 🛠️ System Architecture & Process

This project implements a **Dual-Stage Hybrid Pipeline** designed to tackle the unique challenges of underwater photography: color casting (blue/green tint), low contrast, and turbidity (haze).

### 1. The Physical Model (Variational Brain)
The first stage uses the **Dark Channel Prior (DCP)** principles and the **Underwater Light Attenuation Model**.
*   **Backlight Estimation**: Analyzes the darkest pixels to determine the background water color.
*   **Transmission Mapping**: Generates a depth-intensity heatmap showing how much water is between the camera and the scene.
*   **Scene Radiance Recovery**: Reverses the physical effects of water to produce a "Coarse" restoration.

### 2. The Neural Engine (AI Refinement)
Instead of standard restoration, we use **Residual Learning**.
*   **Residual U-Net**: A 23-layer deep neural network that predicts only the *corrections* needed for the physical model.
*   **Why Residual?**: It ensures the AI doesn't "hallucinate" details, but rather polishes the physics-corrected base.
*   **Loss Function**: A composite blend of **MSE** (color accuracy) and **SSIM** (texture preservation).

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.8+ installed. Run the following to install all dependencies:
```powershell
pip install -r requirements.txt
pip install plotly # For the Premium Dashboard analytics
```

### 2. Dataset Preparation
The system is optimized for the **UIEB Dataset**.
*   Place raw images in: `data/UIEB/train/raw`
*   Place reference images in: `data/UIEB/train/reference`
*   *Note: The system features a Smart Matcher that recursively finds images and ignores corrupted/0-byte files.*

### 3. Training the AI
To start the training process:
```powershell
python train.py --epochs 50 --batch_size 4
```
*   The model saves the best weights automatically to `checkpoints/best_model.pth`.
*   You can stop training anytime and test the latest checkpoint.

### 4. Running the Premium Dashboard
Launch the state-of-the-art web interface:
```powershell
python app.py
```
**Dashboard Features:**
*   **Cyber-Oceanic Light Theme**: Advanced UI for professional presentations.
*   **Performance Analytics**: Real-time graphs showing Latency and MegaPixel throughput.
*   **Pipeline Inspection**: View the internal Transmission Maps and Variational stages.

---

## � Running on Another Device

To migrate this project:
1.  **Transfer**: Copy the project folder (at minimum: `src/`, `app.py`, `requirements.txt`, and `checkpoints/`).
2.  **Setup**: Run `pip install -r requirements.txt`.
3.  **GPU Acceleration**: If the device has an NVIDIA GPU, install the CUDA version of PyTorch for 50x faster training. 
4.  **No-Code Inference**: Simply copy your `best_model.pth` to the new device and run `app.py` to use the pre-trained brain without re-training.

---

## � Performance Profiling
*   **Variational Engine Latency**: ~0.05s - 0.2s depending on resolution.
*   **Neural Refinement Latency**: ~0.1s (GPU) / ~2s (CPU).
*   **Supported Resolutions**: Dynamically scales from 256x256 up to 4K resolution.

---
*Developed for Advanced Underwater Image Research & Development.*
