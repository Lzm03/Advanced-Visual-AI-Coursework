import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import lpips
from torch.utils.data import Dataset

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from torch.utils.data import Dataset

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir=None, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir)) if hr_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        lr_image = Image.open(lr_path).convert("RGB")

        if self.hr_images:
            hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
            hr_image = Image.open(hr_path).convert("RGB")
        else:
            hr_image = lr_image

        lr_image = lr_image.resize((64, 64), Image.BICUBIC)
        hr_image = hr_image.resize((64, 64), Image.BICUBIC)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = DIV2KDataset('./dataset/DIV2K_train_LR_x8', './dataset/DIV2K_train_HR', transform)
val_dataset = DIV2KDataset('./dataset/DIV2K_valid_LR_x8', './dataset/DIV2K_valid_HR', transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.upsample1(x))
        x = self.upsample2(x)
        x = self.conv2(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return torch.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.flatten_size = 128 * 16 * 16 
        self.fc1 = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
    

device = torch.device("mps")
lpips_model = lpips.LPIPS(net='alex').to(device)
torch.backends.cudnn.benchmark = True

generator = Generator().to(device)
discriminator = Discriminator().to(device)

def load_pretrained_model(checkpoint_path, generator, discriminator, optim_G, optim_D):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    # optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
    # optim_D.load_state_dict(checkpoint['optim_D_state_dict'])


def test_on_validation_set(val_loader, generator, save_dir="./results"):
    generator.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    total_samples = 0

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for lr_images, hr_images in val_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            fake_images = generator(lr_images)

            for i in range(hr_images.size(0)):
                hr_image_np = hr_images[i].permute(1, 2, 0).cpu().numpy()
                fake_image_np = fake_images[i].permute(1, 2, 0).cpu().numpy()

                psnr_value = psnr(hr_image_np, fake_image_np, data_range=1.0)
                ssim_value = ssim(hr_image_np, fake_image_np, data_range=1.0, win_size=5, channel_axis=-1)
                lpips_value = lpips_model(hr_images[i].unsqueeze(0), fake_images[i].unsqueeze(0)).item()

                total_psnr += psnr_value
                total_ssim += ssim_value
                total_lpips += lpips_value
                total_samples += 1

                save_image(lr_images[i], os.path.join(save_dir, f"lr_image_{i}.png"))
                save_image(hr_images[i], os.path.join(save_dir, f"hr_image_{i}.png"))
                save_image(fake_images[i], os.path.join(save_dir, f"fake_image_{i}.png"))

    if total_samples > 0:
        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
        avg_lpips = total_lpips / total_samples
        print(f"Validation Results - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
        return avg_psnr, avg_ssim, avg_lpips
    else:
        print("Validation set is empty!")
        return 0.0, 0.0, 0.0


def test_with_noisy_input(val_loader, generator, noise_levels=[5, 50, 100], save_dir="./noisy_results"):
    generator.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    total_samples = 0

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for lr_images, hr_images in val_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            for noise_level in noise_levels:
                noisy_lr_images = lr_images + torch.randn_like(lr_images) * noise_level / 255.0
                noisy_lr_images = noisy_lr_images.clamp(0.0, 1.0)

                fake_images = generator(noisy_lr_images)

                for i in range(hr_images.size(0)):
                    hr_image_np = hr_images[i].permute(1, 2, 0).cpu().numpy()
                    fake_image_np = fake_images[i].permute(1, 2, 0).cpu().numpy()

                    psnr_value = psnr(hr_image_np, fake_image_np, data_range=1.0)
                    ssim_value = ssim(hr_image_np, fake_image_np, data_range=1.0, win_size=5, channel_axis=-1)
                    lpips_value = lpips_model(hr_images[i].unsqueeze(0), fake_images[i].unsqueeze(0)).item()

                    total_psnr += psnr_value
                    total_ssim += ssim_value
                    total_lpips += lpips_value
                    total_samples += 1

                save_image(noisy_lr_images[0], os.path.join(save_dir, f"noisy_lr_{noise_level}_{i}.png"))
                save_image(fake_images[0], os.path.join(save_dir, f"fake_image_{noise_level}_{i}.png"))

    if total_samples > 0:
        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
        avg_lpips = total_lpips / total_samples
        print(f"Noisy Input Results - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
        return avg_psnr, avg_ssim, avg_lpips
    else:
        print("Noisy test set is empty!")
        return 0.0, 0.0, 0.0


def test_with_downscaled_input(val_loader, generator, save_dir="./downscaled_results"):
    generator.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    total_samples = 0

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for lr_images, hr_images in val_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            downscaled_lr_images = F.interpolate(lr_images, scale_factor=0.5, mode='bicubic', align_corners=False)

            fake_images = generator(downscaled_lr_images)

            for i in range(hr_images.size(0)):
                hr_image_np = hr_images[i].permute(1, 2, 0).cpu().numpy()
                fake_image_np = fake_images[i].permute(1, 2, 0).cpu().numpy()

                psnr_value = psnr(hr_image_np, fake_image_np, data_range=1.0)
                ssim_value = ssim(hr_image_np, fake_image_np, data_range=1.0, win_size=5, channel_axis=-1)
                lpips_value = lpips_model(hr_images[i].unsqueeze(0), fake_images[i].unsqueeze(0)).item()

                total_psnr += psnr_value
                total_ssim += ssim_value
                total_lpips += lpips_value
                total_samples += 1

                save_image(downscaled_lr_images[i], os.path.join(save_dir, f"downscaled_lr_{i}.png"))
                save_image(fake_images[i], os.path.join(save_dir, f"fake_image_{i}.png"))

    if total_samples > 0:
        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
        avg_lpips = total_lpips / total_samples
        print(f"Downscaled Input Results - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
        return avg_psnr, avg_ssim, avg_lpips
    else:
        print("Downscaled test set is empty!")
        return 0.0, 0.0, 0.0

optim_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
load_pretrained_model("./model_checkpoints/checkpoint_epoch_23.pth", generator, discriminator, optim_G, optim_D)


test_on_validation_set(val_loader, generator)


test_with_noisy_input(val_loader, generator)


test_with_downscaled_input(val_loader, generator)
