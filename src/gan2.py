import os
import time
import argparse
from multiprocessing import cpu_count
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torch.nn as nn
import lpips 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)
torch.backends.cudnn.benchmark = True

def generate_hr_from_lr(lr_dir, hr_dir, scale=8):
    os.makedirs(hr_dir, exist_ok=True)
    for img_name in os.listdir(lr_dir):
        lr_path = os.path.join(lr_dir, img_name)
        hr_path = os.path.join(hr_dir, img_name)

        lr_image = Image.open(lr_path).convert("RGB")
        
        hr_image = lr_image.resize((lr_image.width * scale, lr_image.height * scale), Image.BICUBIC)
        hr_image.save(hr_path)

generate_hr_from_lr('./dataset/DIV2K_train_LR_x8', './dataset/DIV2K_train_HR', scale=8)
generate_hr_from_lr('./dataset/DIV2K_valid_LR_x8', './dataset/DIV2K_valid_HR', scale=8)

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

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = DIV2KDataset('./dataset/DIV2K_train_LR_x8', './dataset/DIV2K_train_HR', transform)
val_dataset = DIV2KDataset('./dataset/DIV2K_valid_LR_x8', './dataset/DIV2K_valid_HR', transform)



train_dataset = DIV2KDataset(root_dir='./dataset/DIV2K_train_LR_x8', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_dataset = DIV2KDataset(root_dir='./dataset/DIV2K_valid_LR_x8', transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # 像素重排上采样
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res_blocks(x)
        x = self.upsample(x)
        return torch.sigmoid(self.conv2(x))  # 输出范围[0, 1]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)  # 残差连接




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Define the Binary Cross Entropy loss function
loss_func = nn.BCELoss()

# def discriminator_loss(real_output, fake_output):
#     # # Loss for real images
#     # real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
#     # # Loss for fake images
#     # fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))

#     # loss_D = real_loss + fake_loss
    
#     real_labels = torch.ones_like(real_output) * 0.9  # 将真实标签设为0.9而非1
#     fake_labels = torch.zeros_like(fake_output) + 0.1  # 将虚假标签设为0.1而非0

#     real_loss = loss_func(real_output, real_labels.to(device))
#     fake_loss = loss_func(fake_output, fake_labels.to(device))
#     loss_D = real_loss + fake_loss

#     return loss_D

def generator_loss(fake_output, fake_image, real_image):
    # 对抗损失
    adv_loss = loss_func(fake_output, torch.ones_like(fake_output).to(device))
    # 感知损失
    perceptual_loss = F.mse_loss(vgg_extractor(fake_image), vgg_extractor(real_image))
    # 像素级损失
    pixel_loss = F.mse_loss(fake_image, real_image)
    return adv_loss + 0.006 * perceptual_loss + 0.01 * pixel_loss


def discriminator_loss(real_output, fake_output):
    # 真实图像和生成图像的判别损失
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))
    return real_loss + fake_loss

from utils.save_checkpoint import save_checkpoint

def training(x):
    
    '''Training step for the Discriminator'''
    real_x = x.to(device)
    real_output = gan_D(real_x)

    # Backpropagate the discriminator loss and update its parameters
    # for _ in range(2):
    fake_x = gan_G(torch.randn([batch_size, input_dim]).to(device)).detach()
    fake_output = gan_D(fake_x)
    loss_D =discriminator_loss(real_output, fake_output)
    optim_D.zero_grad()
    loss_D.backward()
    optim_D.step()
    
    # if abs(loss_D.item()) < 0.1:  # 判别器和生成器接近平衡
    #     generator_steps = 10
    # elif abs(loss_D.item()) > 0.5:  # 判别器过强
    #     generator_steps = 1
    # else:
    #     generator_steps = 5
        
    '''Training step for the Generator'''
    for _ in range(5):
        fake_x = gan_G(torch.randn([batch_size, input_dim]).to(device))
        fake_output = gan_D(fake_x)
        loss_G = generator_loss(fake_output)

        # Backpropagate the generator loss and update its parameters
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

    return loss_D, loss_G

def discriminator_loss(real_output, fake_output):
    # Loss for real images
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
    # Loss for fake images
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))

    loss_D = real_loss + fake_loss
    return loss_D

def generator_loss(fake_output):
    # Compare discriminator's output on fake images with target labels of 1
    loss_G = loss_func(fake_output, torch.ones_like(fake_output).to(device))
    return loss_G

# task: generate and visualise images at each epoch.
def visualise_generated_images(generator, input_dim, epoch, image_dir, num_images=64):
    """
    Parameters: 
        - num_images: the number of images generated, defaults to 64.
    """ 
    generator.eval()  # evaluation mode
    with torch.no_grad():
        x = torch.randn(num_images, input_dim).to(device)  # random noise
        img = generator(x)  # using the generator to generate images
    save_image(img, f'{image_dir}{epoch + 1}_epoch.png') 
    generator.train()

def display_image(epoch_no):
    # display a single image using the epoch number
    return Image.open(f'{image_dir}{epoch_no}_epoch.png')

def load_checkpoint(checkpoint_path, generator, discriminator, generator_optimizer, discriminator_optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if 'generator_optimizer_state_dict' in checkpoint:
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    else:
        print("Generator optimizer state not found in checkpoint.")
    
    if 'discriminator_optimizer_state_dict' in checkpoint:
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
    else:
        print("Discriminator optimizer state not found in checkpoint.")
    
    start_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch + 1}.")
    return start_epoch

def validate(generator, dataloader, epoch, image_dir):
    generator.eval()
    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for lr_images, hr_images in dataloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            # 使用生成器生成高分辨率图像
            sr_images = generator(lr_images)

            # 计算指标
            for i in range(hr_images.size(0)):
                psnr_value = psnr(
                    hr_images[i].cpu().numpy(),
                    sr_images[i].cpu().numpy(),
                    data_range=1.0
                )
                ssim_value = ssim(
                    hr_images[i].permute(1, 2, 0).cpu().numpy(),
                    sr_images[i].permute(1, 2, 0).cpu().numpy(),
                    multichannel=True
                )
                total_psnr += psnr_value
                total_ssim += ssim_value

            lpips_value = lpips_model(hr_images, sr_images).mean().item()
            total_lpips += lpips_value
            total_samples += hr_images.size(0)

    print(f"Validation - Epoch {epoch + 1}:")
    print(f"PSNR: {total_psnr / total_samples:.4f}, SSIM: {total_ssim / total_samples:.4f}, LPIPS: {total_lpips / total_samples:.4f}")



# task: visualise the loss from the training part
def visualise_loss(losses_D, losses_G, image_dir, loss_type):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_D, label='Discriminator Loss')
    plt.plot(losses_G, label='Generator Loss')
    plt.title('Training Loss')
    plt.xlabel(f'{loss_type}')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid()
    plt.savefig(f'{image_dir}/training_loss_{loss_type}.png')  # save the loss plot
    plt.show()
    plt.close()

    
input_dim = 100
batch_size = 64
num_epoch = 50
lr = 0.0001
checkpoint_dir = './training_checkpoints_29'
os.makedirs(checkpoint_dir, exist_ok=True)
image_dir = './generated_image_29/'
os.makedirs(image_dir, exist_ok=True)

gan_G = Generator(input_dim).to(device)
gan_D = Discriminator().to(device)
optim_D = torch.optim.Adam(gan_D.parameters(), lr=lr, betas=(0.5, 0.999))
optim_G = torch.optim.Adam(gan_G.parameters(), lr=lr, betas=(0.5, 0.999))
# lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optim_G, gamma=1.5)
# lr_scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=1.5)
# optim_G = torch.optim.Adam(gan_G.parameters(), lr=0.0008, betas=(0.5, 0.999))
# optim_D = torch.optim.Adam(gan_D.parameters(), lr=0.000001, betas=(0.5, 0.999))
# scheduler_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=10, gamma=0.5)
# scheduler_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=10, gamma=0.5)
checkpoint_path = './training_checkpoints_28/ckpt_epoch_0018.pth'

start_epoch = 0
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(checkpoint_path, gan_G, gan_D, optim_G, optim_D)
else:
    print("No checkpoint found. Starting training from scratch.")

iteration_losses_D = []
iteration_losses_G = []
epoch_losses_D = []
epoch_losses_G = []

for epoch in range(num_epoch):
    start_time = time.time()
    total_loss_D, total_loss_G = 0, 0
    
    for i, (x, _) in enumerate(train_loader):
        loss_D, loss_G = training(x)

        iteration_losses_D.append(loss_D.detach().item())
        iteration_losses_G.append(loss_G.detach().item())
        total_loss_D += loss_D.detach().item()
        total_loss_G += loss_G.detach().item()
        
    # lr_scheduler_G.step()
    # lr_scheduler_D.step()
        
    epoch_losses_D.append(total_loss_D / len(train_loader))
    epoch_losses_G.append(total_loss_G / len(train_loader))
    
    # Save model checkpoints
    if (epoch + 1) % 2 == 0:
        save_checkpoint(epoch + 1, gan_G, gan_D, optim_G, optim_D, checkpoint_dir)

    # losses once per epoch
    print(f'Epoch [{epoch + 1}/{num_epoch}] | Loss_D {iteration_losses_D[-1]:.4f} | Loss_G {iteration_losses_G[-1]:.4f} | Time: {time.time() - start_time:.2f} sec')
    print(f'Epoch [{epoch + 1}/{num_epoch}]  | Loss_D {epoch_losses_D[epoch]:.4f} | Loss_G {epoch_losses_G[epoch]:.4f} | Time: {time.time() - start_time:.2f} sec')
    
    validate(gan_G, val_loader, epoch, image_dir)
    # Task1: visualise the generated image at different epochs
    visualise_generated_images(gan_G, input_dim, epoch, image_dir)

    
# Task2: visualise the loss through a plot
visualise_loss(iteration_losses_D, iteration_losses_G, image_dir, 'Iteration')
visualise_loss(epoch_losses_D, epoch_losses_G, image_dir, 'Epoch')