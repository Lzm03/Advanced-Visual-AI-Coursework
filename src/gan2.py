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

device = torch.device("cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)
torch.backends.cudnn.benchmark = True

# def generate_hr_from_lr(lr_dir, hr_dir, scale=8):
#     os.makedirs(hr_dir, exist_ok=True)
#     for img_name in os.listdir(lr_dir):
#         lr_path = os.path.join(lr_dir, img_name)
#         hr_path = os.path.join(hr_dir, img_name)

#         lr_image = Image.open(lr_path).convert("RGB")
#         hr_image = lr_image.resize((lr_image.width * scale, lr_image.height * scale), Image.BICUBIC)
#         hr_image.save(hr_path)

# generate_hr_from_lr('./dataset/DIV2K_train_LR_x8', './dataset/DIV2K_train_HR', scale=8)
# generate_hr_from_lr('./dataset/DIV2K_valid_LR_x8', './dataset/DIV2K_valid_HR', scale=8)

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

        # 动态调整图像大小
        lr_image = lr_image.resize((64, 64), Image.BICUBIC)
        hr_image = hr_image.resize((64, 64), Image.BICUBIC)

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image


transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小到 64x64
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
        x = torch.sigmoid(x)  # 将输出限制在 [0, 1]
        return x


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
        self.fc1 = None  # 延迟初始化
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.fc1 is None:
            flatten_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Sequential(
                nn.Linear(flatten_size, 1024),
                nn.LeakyReLU(0.2)
            )
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)





loss_func = nn.BCELoss()

def generator_loss(fake_output, fake_image, real_image):
    adv_loss = loss_func(fake_output, torch.ones_like(fake_output).to(device))
    pixel_loss = F.mse_loss(fake_image, real_image)
    
    return adv_loss + 0.01 * pixel_loss

def discriminator_loss(real_output, fake_output):
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))

    return real_loss + fake_loss

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optim_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

import torch
from tqdm import tqdm

for epoch in range(10):
    generator.train()
    discriminator.train()
    total_loss_D = 0
    total_loss_G = 0
    
    for lr_images, hr_images in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        fake_images = generator(lr_images)
        real_output = discriminator(hr_images)
        fake_output = discriminator(fake_images.detach())
        loss_D = discriminator_loss(real_output, fake_output)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        fake_output = discriminator(fake_images)
        loss_G = generator_loss(fake_output, fake_images, hr_images)
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

    print(f"Epoch {epoch+1}, Loss_D: {total_loss_D/len(train_loader):.4f}, Loss_G: {total_loss_G/len(train_loader):.4f}")
