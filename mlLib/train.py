import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from PIL import Image
import os
import random

IMG_WIDTH = 800
IMG_HEIGHT = 600
PATH = "/media/user/2TB Clear/imageData"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS_PER_CYCLE = 10
NUM_CYCLES = 10
SAMPLES_PER_CYCLE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImagePairDataset(Dataset):
    def __init__(self, folder_list, width=800, height=600):
        self.folders = folder_list
        self.width = width
        self.height = height
        
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, idx):
        folder = self.folders[idx]
        
        low_res = Image.open(os.path.join(folder, "low_res.png")).resize((self.width, self.height))
        high_res = Image.open(os.path.join(folder, "high_res.png")).resize((self.width, self.height))
        
        # Convert grayscale to RGB if needed
        if low_res.mode != 'RGB':
            low_res = low_res.convert('RGB')
        if high_res.mode != 'RGB':
            high_res = high_res.convert('RGB')
        
        low_res = torch.from_numpy(np.array(low_res, dtype=np.float32)).permute(2, 0, 1) / 255.0
        high_res = torch.from_numpy(np.array(high_res, dtype=np.float32)).permute(2, 0, 1) / 255.0
        
        return low_res, high_res

class DirectPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(16, 3, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        b = self.bottleneck(e4)
        
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        output = self.output(d1)
        
        # Ensure output matches input size exactly
        if output.shape[2:] != input_size:
            output = nn.functional.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        return output

class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-6)
        target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-6)
        
        dot = (pred_norm * target_norm).sum(dim=1)
        direction_loss = (1.0 - dot).mean()
        
        pred_sat = pred.std(dim=1)
        target_sat = target.std(dim=1)
        saturation_loss = torch.clamp(target_sat - pred_sat, min=0).mean()
        
        total = direction_loss + (saturation_loss + 1) ** 3
        
        return total, direction_loss, saturation_loss

class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.15, mae_weight=0.35, color_weight=0.50):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.color_loss = ColorLoss()
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.color_weight = color_weight
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        color_total, color_dir, color_sat = self.color_loss(pred, target)
        
        total = (self.mse_weight * mse_loss + 
                self.mae_weight * mae_loss + 
                self.color_weight * color_total)
        
        return total, {
            'mse': mse_loss.item(),
            'mae': mae_loss.item(),
            'color_total': color_total.item(),
            'color_direction': color_dir.item(),
            'color_saturation': color_sat.item()
        }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics = {'mse': 0, 'mae': 0, 'color_total': 0, 'color_direction': 0, 'color_saturation': 0}
    
    for low_res, high_res in dataloader:
        low_res = low_res.to(device)
        high_res = high_res.to(device)
        
        optimizer.zero_grad()
        pred = model(low_res)
        loss, loss_dict = criterion(pred, high_res)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for key in metrics:
            metrics[key] += loss_dict[key]
    
    num_batches = len(dataloader)
    avg_metrics = {k: v / num_batches for k, v in metrics.items()}
    avg_metrics['total_loss'] = total_loss / num_batches
    
    return avg_metrics

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'mse': 0, 'mae': 0, 'color_total': 0, 'color_direction': 0, 'color_saturation': 0}
    
    with torch.no_grad():
        for low_res, high_res in dataloader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            pred = model(low_res)
            loss, loss_dict = criterion(pred, high_res)
            
            total_loss += loss.item()
            for key in metrics:
                metrics[key] += loss_dict[key]
    
    num_batches = len(dataloader)
    avg_metrics = {k: v / num_batches for k, v in metrics.items()}
    avg_metrics['total_loss'] = total_loss / num_batches
    
    return avg_metrics

def generate_samples(model, dataloader, device, num_samples=4):
    model.eval()
    samples = []
    
    with torch.no_grad():
        for low_res, high_res in dataloader:
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            pred = model(low_res)
            
            # Take only the requested number of samples
            batch_size = min(num_samples - len(samples), low_res.shape[0])
            
            for i in range(batch_size):
                # Convert tensors to numpy and transpose to HWC format
                input_img = low_res[i].cpu().permute(1, 2, 0).numpy()
                target_img = high_res[i].cpu().permute(1, 2, 0).numpy()
                pred_img = pred[i].cpu().permute(1, 2, 0).numpy()
                
                # Clip to valid range
                input_img = np.clip(input_img, 0, 1)
                target_img = np.clip(target_img, 0, 1)
                pred_img = np.clip(pred_img, 0, 1)
                
                samples.append({
                    'input': wandb.Image(input_img, caption='Input (Noisy)'),
                    'target': wandb.Image(target_img, caption='Target (Clean)'),
                    'prediction': wandb.Image(pred_img, caption='Prediction')
                })
            
            if len(samples) >= num_samples:
                break
    
    return samples

def main():
    wandb.init(
        project="color-restoration-teacher",
        config={
            "architecture": "DirectPrediction-10M",
            "img_width": IMG_WIDTH,
            "img_height": IMG_HEIGHT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs_per_cycle": EPOCHS_PER_CYCLE,
            "num_cycles": NUM_CYCLES,
            "samples_per_cycle": SAMPLES_PER_CYCLE,
            "mse_weight": 0.4,
            "mae_weight": 0.5,
            "color_weight": 0.10
        }
    )
    
    all_folders = [f.path for f in os.scandir(PATH) if f.is_dir()]
    print(f"Total folders: {len(all_folders)}")
    
    model = DirectPredictionModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    criterion = CombinedLoss(mse_weight=0.15, mae_weight=0.35, color_weight=0.50)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    global_epoch = 0
    
    for cycle in range(NUM_CYCLES):
        print(f"\nCycle {cycle+1}/{NUM_CYCLES}")
        
        selected_folders = random.sample(all_folders, min(SAMPLES_PER_CYCLE, len(all_folders)))
        
        train_size = int(0.8 * len(selected_folders))
        train_folders = selected_folders[:train_size]
        val_folders = selected_folders[train_size:]
        
        train_dataset = ImagePairDataset(train_folders, IMG_WIDTH, IMG_HEIGHT)
        val_dataset = ImagePairDataset(val_folders, IMG_WIDTH, IMG_HEIGHT)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        for epoch in range(EPOCHS_PER_CYCLE):
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_metrics = validate(model, val_loader, criterion, DEVICE)
            
            # Generate sample images for visualization
            samples = generate_samples(model, val_loader, DEVICE, num_samples=4)
            
            # Prepare wandb log data
            log_data = {
                'cycle': cycle,
                'global_epoch': global_epoch,
                'train_loss': train_metrics['total_loss'],
                'train_mse': train_metrics['mse'],
                'train_mae': train_metrics['mae'],
                'train_color': train_metrics['color_total'],
                'val_loss': val_metrics['total_loss'],
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_color': val_metrics['color_total']
            }
            
            # Add sample images to log
            for idx, sample in enumerate(samples):
                log_data[f'sample_{idx}_input'] = sample['input']
                log_data[f'sample_{idx}_target'] = sample['target']
                log_data[f'sample_{idx}_prediction'] = sample['prediction']
            
            wandb.log(log_data)
            
            print(f"Epoch {epoch+1} - Train: {train_metrics['total_loss']:.4f}, Val: {val_metrics['total_loss']:.4f}")
            
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save({
                    'cycle': cycle,
                    'epoch': epoch,
                    'global_epoch': global_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, 'teacher_model.pth')
                print(f"Saved: {best_val_loss:.4f}")
            
            global_epoch += 1
    
    wandb.finish()
    print("Complete")

if __name__ == "__main__":
    main()