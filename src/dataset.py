import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .variational import VariationalEnhancer

class UnderwaterDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, size=(256, 256), transform=None):
        """
        Args:
            raw_dir (str): Directory with raw underwater images.
            ref_dir (str): Directory with reference (ground truth) images.
            size (tuple): Resize dimensions (H, W).
            transform: Optional transforms.
        """
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.size = size
        self.transform = transform
        
        # Filter matching files recursively
        def find_images(directory):
            found = []
            for root, _, files in os.walk(directory):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Store relative path from the base directory
                        rel_path = os.path.relpath(os.path.join(root, f), directory)
                        # Filter out 0-byte or corrupted looking files if necessary
                        if os.path.getsize(os.path.join(root, f)) > 0:
                            found.append(rel_path)
            return found

        self.raw_image_paths = find_images(raw_dir)
        self.ref_image_paths = find_images(ref_dir)
        
        # Create mapping by matching filename stems
        # Map stems to their relative paths
        raw_map = {os.path.splitext(os.path.basename(p))[0].strip(): p for p in self.raw_image_paths}
        ref_map = {os.path.splitext(os.path.basename(p))[0].strip(): p for p in self.ref_image_paths}
        
        self.pairs = []
        for stem, raw_rel in raw_map.items():
            if stem in ref_map:
                self.pairs.append((raw_rel, ref_map[stem]))
            else:
                # Fuzzy matching for stems with " - Copy" or similar
                clean_stem = stem.replace(" - Copy", "").strip()
                match = next((s for s in ref_map if clean_stem in s or s in clean_stem), None)
                if match:
                    self.pairs.append((raw_rel, ref_map[match]))
        
        
        # Validate pairs - remove corrupted images
        valid_pairs = []
        corrupted_count = 0
        
        for i, (raw_name, ref_name) in enumerate(self.pairs):
            raw_path = os.path.abspath(os.path.join(raw_dir, raw_name))
            ref_path = os.path.abspath(os.path.join(ref_dir, ref_name))
            
            raw_test = self.robust_read(raw_path)
            ref_test = self.robust_read(ref_path)
            
            if raw_test is not None and ref_test is not None:
                valid_pairs.append((raw_name, ref_name))
            else:
                corrupted_count += 1
                if corrupted_count <= 5: # Only print first 5 to avoid spam
                    if not os.path.exists(raw_path):
                        print(f"❌ File not found: {raw_path}")
                    elif raw_test is None:
                        print(f"❌ OpenCV failed to decode: {raw_path}")
                    
                    if not os.path.exists(ref_path):
                        print(f"❌ File not found: {ref_path}")
                    elif ref_test is None:
                        print(f"❌ OpenCV failed to decode: {ref_path}")
                
        self.pairs = valid_pairs
        self.enhancer = VariationalEnhancer()
        print(f"Found {len(self.pairs)} valid paired images.")
        if corrupted_count > 0:
            print(f"⚠️  Skipped {corrupted_count} corrupted/unreadable images.")

    def __len__(self):
        return len(self.pairs)

    def robust_read(self, path):
        if not os.path.exists(path):
            return None
        try:
            img_array = np.fromfile(path, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except:
            return None

    def __getitem__(self, idx):
        raw_name, ref_name = self.pairs[idx]
        
        raw_path = os.path.abspath(os.path.join(self.raw_dir, raw_name))
        ref_path = os.path.abspath(os.path.join(self.ref_dir, ref_name))
        
        # Read images
        raw_bgr = self.robust_read(raw_path)
        ref_bgr = self.robust_read(ref_path)
        
        if raw_bgr is None or ref_bgr is None:
            # This should never happen since we validate during init, but just in case
            print(f"Warning: Skipping corrupted image at runtime: {raw_name}")
            # Return a black image as fallback
            black_img = np.zeros((self.size[1], self.size[0], 3), dtype=np.float32)
            black_tensor = torch.from_numpy(black_img.transpose((2, 0, 1)))
            return black_tensor, black_tensor

        # Variational Preprocessing (Physics-based enhancement)
        # This acts as the "Coarse" restoration
        variational_out, _ = self.enhancer.process(image_numpy=raw_bgr)
        
        # Convert Ref to 0-1 RGB
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Resize
        # Note: Variational output is already RGB float 0-1 from enhancer
        # resizing numpy arrays
        variational_out = cv2.resize(variational_out, self.size)
        ref_rgb = cv2.resize(ref_rgb, self.size)
        
        # To Tensor (C, H, W)
        input_tensor = torch.from_numpy(variational_out.transpose((2, 0, 1)))
        target_tensor = torch.from_numpy(ref_rgb.transpose((2, 0, 1)))
        
        return input_tensor, target_tensor
