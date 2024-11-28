import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

lpips_model = lpips.LPIPS(net='alex').to(torch.device("cpu"))

def calculate_metrics(pred, target):
    pred = torch.clamp(pred, -1, 1).cpu().numpy().transpose(0, 2, 3, 1)
    target = torch.clamp(target, -1, 1).cpu().numpy().transpose(0, 2, 3, 1)

    total_psnr, total_ssim, total_lpips = 0, 0, 0

    for p, t in zip(pred, target):
        p = np.clip(p, -1, 1)
        t = np.clip(t, -1, 1)

        total_psnr += psnr(p, t, data_range=2) 
        total_ssim += ssim(p, t, data_range=2, multichannel=True,win_size=3)
        
    pred_tensor = torch.from_numpy(pred).permute(0, 3, 1, 2).to(torch.device("cpu"))
    target_tensor = torch.from_numpy(target).permute(0, 3, 1, 2).to(torch.device("cpu"))
    total_lpips = lpips_model(pred_tensor, target_tensor).mean().item()

    avg_psnr = total_psnr / len(pred)
    avg_ssim = total_ssim / len(pred)
    return avg_psnr, avg_ssim, total_lpips

def evaluate_dip_model(model, valid_loader):
    model.eval()
    total_psnr, total_ssim, total_lpips = 0, 0, 0

    with torch.no_grad():
        for lr_images, hr_images in valid_loader:
            output = model(lr_images)

            avg_psnr, avg_ssim, avg_lpips = calculate_metrics(output, hr_images)
            total_psnr += avg_psnr
            total_ssim += avg_ssim
            total_lpips += avg_lpips

    avg_psnr = total_psnr / len(valid_loader)
    avg_ssim = total_ssim / len(valid_loader)
    avg_lpips = total_lpips / len(valid_loader)

    print(f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}")



class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.upsample(x) 
        return x


class SuperResolutionDataset(torch.utils.data.Dataset):
    def __init__(self, lr_folder, hr_folder, transform=None):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.lr_images = sorted(os.listdir(lr_folder))
        self.hr_images = sorted(os.listdir(hr_folder))
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image_path = os.path.join(self.lr_folder, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_folder, self.hr_images[idx])

        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = SuperResolutionDataset('../dataset/DIV2K_train_LR_x8/', '../dataset/DIV2K_train_HR/', transform=transform)
# valid_dataset = SuperResolutionDataset('../dataset/DIV2K_valid_LR_x8/', '../dataset/DIV2K_valid_HR/', transform=transform)
# valid_dataset = SuperResolutionDataset('../dataset/noise_images_100_LR_x8', '../dataset/DIV2K_valid_HR/', transform=transform)
valid_dataset = SuperResolutionDataset('../dataset/LR_x16', '../dataset/DIV2K_valid_HR/', transform=transform)


def get_subset(dataset, subset_size=1000):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)


train_subset = get_subset(train_dataset, subset_size=20)
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)


def loss_fn(pred, target):
    return nn.MSELoss()(pred, target)


def train_dip_model(model, optimizer, train_loader, num_epochs=10, checkpoint_interval=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for lr_images, hr_images in train_loader:
            optimizer.zero_grad()
            output = model(lr_images)
            loss = loss_fn(output, hr_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1)

    return model


def visualize_results(model, valid_loader):
    model.eval()
    with torch.no_grad():
        lr_images, hr_images = next(iter(valid_loader))
        # lr_images, hr_images = lr_images.cuda(), hr_images.cuda()
        output = model(lr_images)
        index = 4
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(lr_images[index].cpu().permute(1, 2, 0).numpy())
        axs[0].set_title('Low Resolution')
        axs[1].imshow(output[index].cpu().permute(1, 2, 0).numpy())
        axs[1].set_title('DIP Model Output')
        axs[2].imshow(hr_images[index].cpu().permute(1, 2, 0).numpy())
        axs[2].set_title('High Resolution')
        plt.show()

def initialize_dip_model():
    model = DIPModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer

def save_checkpoint(model, optimizer, epoch, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"dip_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model weights loaded from {checkpoint_path}")
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded.")
    
    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded: Resuming from epoch {epoch}")
    
    return model, optimizer, epoch



if __name__ == '__main__':
    model, optimizer = initialize_dip_model()

    checkpoint_path = "./checkpoints/dip_epoch_1990.pth"

    start_epoch = 0 
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    # trained_model = train_dip_model(model, optimizer, train_loader, num_epochs=2000 - start_epoch)

    # evaluate_dip_model(model, valid_loader)
    visualize_results(model, valid_loader)