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

device = torch.device("cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)
torch.backends.cudnn.benchmark = True


# 数据集定义
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
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = DIV2KDataset('./dataset/DIV2K_train_LR_x8', './dataset/DIV2K_train_HR', transform)
val_dataset = DIV2KDataset('./dataset/DIV2K_valid_LR_x8', './dataset/DIV2K_valid_HR', transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)


# 模型定义
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



# 损失函数定义
loss_func = nn.BCELoss()

def generator_loss(fake_output, fake_image, real_image):
    adv_loss = loss_func(fake_output, torch.ones_like(fake_output).to(device))
    pixel_loss = F.mse_loss(fake_image, real_image)
    return adv_loss + 0.01 * pixel_loss


def discriminator_loss(real_output, fake_output):
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))
    return real_loss + fake_loss


# 保存和加载模型
def save_model(generator, discriminator, epoch, save_dir="./model_checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optim_G_state_dict': optim_G.state_dict(),
        'optim_D_state_dict': optim_D.state_dict()
    }, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
    print(f"Model checkpoint saved for epoch {epoch}.")


def load_model(checkpoint_path, generator, discriminator, optim_G, optim_D):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    discriminator_state_dict = checkpoint['discriminator_state_dict']
    model_state_dict = discriminator.state_dict()
    updated_state_dict = {k: v for k, v in discriminator_state_dict.items() if k in model_state_dict}
    model_state_dict.update(updated_state_dict)
    discriminator.load_state_dict(model_state_dict)

    optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
    optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
    return checkpoint['epoch']



def validate(generator, val_loader):
    generator.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for lr_images, hr_images in val_loader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)
            fake_images = generator(lr_images)

            for i in range(hr_images.size(0)):
                # 转换为 numpy 格式
                hr_image_np = hr_images[i].permute(1, 2, 0).cpu().numpy()
                fake_image_np = fake_images[i].permute(1, 2, 0).cpu().numpy()

                # 计算 PSNR
                psnr_value = psnr(hr_image_np, fake_image_np, data_range=1.0)

                # 计算 SSIM，显式指定 data_range
                ssim_value = ssim(
                    hr_image_np,
                    fake_image_np,
                    data_range=1.0,  # 显式指定值域为 1.0
                    win_size=5,      # 窗口大小为 5
                    channel_axis=-1  # 指定颜色通道轴
                )

                lpips_value = lpips_model(
                    hr_images[i].unsqueeze(0),
                    fake_images[i].unsqueeze(0)
                ).item()

                total_psnr += psnr_value
                total_ssim += ssim_value
                total_lpips += lpips_value
                total_samples += 1

    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_lpips = total_lpips / total_samples

    print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")



generator = Generator().to(device)
discriminator = Discriminator().to(device)

optim_G = Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optim_D = Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

def save_images(epoch, lr_images, hr_images, fake_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (lr, hr, fake) in enumerate(zip(lr_images, hr_images, fake_images)):
        save_image(lr, os.path.join(save_dir, f"epoch_{epoch}_lr_{i}.png"))
        save_image(hr, os.path.join(save_dir, f"epoch_{epoch}_hr_{i}.png"))
        save_image(fake, os.path.join(save_dir, f"epoch_{epoch}_fake_{i}.png"))



# start_epoch = load_model("./model_checkpoints/checkpoint_epoch.pth", generator, discriminator, optim_G, optim_D)

start_epoch = 0
results_dir = "./training_result"

for epoch in range(start_epoch, 50):
    generator.train()
    discriminator.train()
    total_loss_D, total_loss_G = 0.0, 0.0

    for batch_idx, (lr_images, hr_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        # 判别器训练
        fake_images = generator(lr_images)
        real_output = discriminator(hr_images)
        fake_output = discriminator(fake_images.detach())
        loss_D = discriminator_loss(real_output, fake_output)
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # 生成器训练
        fake_output = discriminator(fake_images)
        loss_G = generator_loss(fake_output, fake_images, hr_images)
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

        # 每个 batch 保存一次图像
        if batch_idx == 0:  # 只保存第一个 batch 的图像
            save_dir = os.path.join(results_dir, f"epoch_{epoch + 1}")
            save_images(epoch + 1, lr_images.cpu(), hr_images.cpu(), fake_images.cpu(), save_dir)

    print(f"Epoch {epoch + 1}, Loss_D: {total_loss_D / len(train_loader):.4f}, Loss_G: {total_loss_G / len(train_loader):.4f}")

    # 保存模型
    if (epoch + 1) % 1 == 0:
        save_model(generator, discriminator, epoch + 1)

    # 验证
    validate(generator, val_loader)
