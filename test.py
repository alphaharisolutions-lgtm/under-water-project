import argparse
import os
import torch
import cv2
import numpy as np
from src.model import ResidualUNet
from src.variational import VariationalEnhancer
from src.utils import get_device, save_image_tensor

def enhance_single_image(args):
    device = get_device()
    
    # Load Model
    model = ResidualUNet(n_channels=3, n_classes=3).to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights (Result will be poor).")
    
    model.eval()

    # Read Image
    if not os.path.exists(args.input):
        print(f"Error: Input image {args.input} not found.")
        return

    raw_bgr = cv2.imread(args.input)
    
    # 1. Variational Preprocessing
    print("Applying Variational Enhancement...")
    enhancer = VariationalEnhancer()
    variational_out, _ = enhancer.process(image_numpy=raw_bgr)
    
    # Resize for Model (and keep original size to resize back if needed)
    # For U-Net, dimensions usually need to be divisible by 16 or 32.
    # We resize to 256x256 for simplicity as per training.
    # In a production app, we might tile or resize smartly.
    
    orig_h, orig_w = variational_out.shape[:2]
    input_resized = cv2.resize(variational_out, (256, 256))
    
    # To Tensor
    input_tensor = torch.from_numpy(input_resized.transpose((2, 0, 1))).unsqueeze(0).float().to(device)
    
    # 2. Model Inference
    print("Running Machine Learning Refinement...")
    # Calculate residual at low-res
    with torch.no_grad():
        output_low_res = model(input_tensor)
        # Residual = Output - Input (at 256x256)
        residual_low_res = output_low_res - input_tensor
    
    # 3. High-Fidelity Reconstruction
    # Resize the residual back to full resolution
    residual_np = residual_low_res.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    residual_high_res = cv2.resize(residual_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    # Apply residual to full-res variational output
    final_full_res = variational_out + residual_high_res
    final_full_res = np.clip(final_full_res, 0, 1)
    
    # Save Results
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.input)
    
    # Save Variational
    var_path = os.path.join(args.output_dir, f"variational_{base_name}")
    var_bgr = cv2.cvtColor((variational_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(var_path, var_bgr)
    
    # Save Final Output (Full Resolution)
    out_path = os.path.join(args.output_dir, f"final_{base_name}")
    # Convert to BGR uint8
    final_bgr = cv2.cvtColor((final_full_res * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Optional final sharpening for clarity
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    final_bgr = cv2.filter2D(final_bgr, -1, kernel)
    
    cv2.imwrite(out_path, final_bgr)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input underwater image (optional, will pick sample if omitted)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # If no input provided, try to find a sample in the data directory
    if not args.input:
        sample_dir = 'data/UIEB/test/raw'
        if os.path.exists(sample_dir):
            files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                args.input = os.path.join(sample_dir, files[0])
                print(f"No input specified. Using sample: {args.input}")
            else:
                print("Error: No images found in data/UIEB/test/raw. Please specify --input.")
                exit(1)
        else:
            print("Error: Please specify --input <path_to_image>.")
            exit(1)
            
    enhance_single_image(args)
