import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from src.dataset import UnderwaterDataset
from src.model import ResidualUNet
from src.losses import CompositeLoss
from src.utils import get_device

def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset & Loader
    dataset = UnderwaterDataset(
        raw_dir=os.path.join(args.data_dir, 'train/raw'),
        ref_dir=os.path.join(args.data_dir, 'train/reference'),
        size=(256, 256)
    )
    
    if len(dataset) == 0:
        print("No images found! Check your data directory structure.")
        print(f"Expected: {args.data_dir}/train/raw and {args.data_dir}/train/reference")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = ResidualUNet(n_channels=3, n_classes=3).to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CompositeLoss(alpha=0.5).to(device)

    print("Starting Training...")
    
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(loader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(1)

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print("Saved Best Model!")

        # Save Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'ckpt_epoch_{epoch+1}.pth'))

    print("Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/UIEB', help='Root dataset directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    train(args)
