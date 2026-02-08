import os
import shutil
import glob

def split_dataset(val_count=90):
    base_dir = "data/UIEB"
    
    # Source directories (where user put everything)
    train_raw = os.path.join(base_dir, "train", "raw")
    train_ref = os.path.join(base_dir, "train", "reference")
    
    # Destination directories (test set)
    test_raw = os.path.join(base_dir, "test", "raw")
    test_ref = os.path.join(base_dir, "test", "reference")
    
    # Ensure they exist
    os.makedirs(test_raw, exist_ok=True)
    os.makedirs(test_ref, exist_ok=True)
    
    # Get list of files
    raw_files = sorted(glob.glob(os.path.join(train_raw, "*")))
    ref_files = sorted(glob.glob(os.path.join(train_ref, "*")))
    
    if len(raw_files) == 0:
        print("No files found in data/UIEB/train/raw. Please paste images first!")
        return

    # Check matching counts
    if len(raw_files) != len(ref_files):
        print("Warning: Number of raw and reference images do not match!")
        print(f"Raw: {len(raw_files)}, Ref: {len(ref_files)}")
    
    print(f"Found {len(raw_files)} images total.")
    print(f"Moving last {val_count} images to Test folder...")
    
    # Move last 'val_count' files
    files_to_move_raw = raw_files[-val_count:]
    
    count = 0
    for f_path in files_to_move_raw:
        frame_name = os.path.basename(f_path)
        
        # Move Raw
        shutil.move(f_path, os.path.join(test_raw, frame_name))
        
        # Move corresponding Reference
        ref_path = os.path.join(train_ref, frame_name)
        if os.path.exists(ref_path):
            shutil.move(ref_path, os.path.join(test_ref, frame_name))
            count += 1
        else:
            print(f"Warning: Reference for {frame_name} not found.")

    print(f"Successfully moved {count} pairs to 'data/UIEB/test/'.")
    print("Dataset ready for training!")

if __name__ == "__main__":
    split_dataset()
