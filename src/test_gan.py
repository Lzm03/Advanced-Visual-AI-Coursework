import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips  # LPIPS library
import numpy as np

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集定义
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256 * 8 * 8)  # 原来的 "fc" 改为 "fc1"
        self.br1 = nn.Sequential(  # 原来的 "br" 改为 "br1"
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(  # 原来的 "deconv_blocks" 改为分层命名
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)  # 改为 fc1
        x = self.br1(x)  # 改为 br1
        x = x.view(-1, 256, 8, 8)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


import torch
import numpy as np

def add_noise(image, sigma):
    # Check if the image is a tensor or NumPy array and convert to float32 if needed
    if isinstance(image, torch.Tensor):
        image = image.float()  # Use .float() for PyTorch tensors
    else:
        image = image.astype(np.float32)  # Use astype for NumPy arrays
    
    noise = torch.randn_like(image) * sigma
    noisy_image = image + noise

    # Ensure the noisy image is clipped to the valid range [0, 1]
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image

from skimage.transform import resize

def validate_with_noise(generator, loader, input_dim, image_dir, lpips_fn, noise_levels):
    generator.eval()
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    count = 0

    with torch.no_grad():
        for i, (real_images, _) in enumerate(tqdm(loader, desc="Validating with Noise")):
            real_images = real_images.to(device)

            # 生成噪声并生成图像
            noise = torch.randn(real_images.size(0), input_dim).to(device)
            fake_images = generator(noise)

            for sigma in noise_levels:
                # 添加不同级别的噪声
                noisy_images = [add_noise(image, sigma) for image in real_images]

                for j in range(real_images.size(0)):
                    real = noisy_images[j]
                    fake = fake_images[j].cpu().permute(1, 2, 0).numpy()

                    # Resize real image to match fake image dimensions
                    real_resized = resize(real.cpu().permute(1, 2, 0).numpy(), fake.shape, mode='reflect', anti_aliasing=True)

                    # Convert resized real image to tensor
                    real_tensor = torch.tensor(real_resized).permute(2, 0, 1).to(device).float()

                    # 计算 PSNR 和 SSIM
                    psnr_val = psnr(real_resized, fake, data_range=1.0)
                    ssim_val = ssim(real_resized, fake, data_range=1.0, multichannel=True, win_size=3)

                    # LPIPS input should be normalized to [-1, 1]
                    lpips_val = lpips_fn(
                        (real_tensor.unsqueeze(0) * 2 - 1),  # Convert to [-1, 1]
                        fake_images[j].unsqueeze(0) * 2 - 1,
                    ).item()

                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    total_lpips += lpips_val
                    count += 1

                    # 保存恢复的图像
                    save_image(fake_images[j], os.path.join(image_dir, f"validation_noise_{sigma}_{i}_{j}_generated.png"))
                    save_image(torch.tensor(real_resized).permute(2, 0, 1), os.path.join(image_dir, f"validation_noise_{sigma}_{i}_{j}_noisy.png"))

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count

    print(f"Validation Results with Noise (sigma={sigma}): PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}, LPIPS = {avg_lpips:.4f}")




def validate_with_downscaling(generator, loader, input_dim, image_dir, lpips_fn, scale_factor):
    generator.eval()
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    count = 0

    with torch.no_grad():
        for i, (real_images, _) in enumerate(tqdm(loader, desc="Validating with Downscaling")):
            real_images = real_images.to(device).float()  # Ensure real_images are float32

            noise = torch.randn(real_images.size(0), input_dim).to(device).float()  # Ensure noise is float32
            fake_images = generator(noise)
            downscaled_images = [downscale_image(image, scale_factor) for image in real_images]

            for j in range(real_images.size(0)):
                real = downscaled_images[j].cpu().permute(1, 2, 0).numpy()
                fake = fake_images[j].cpu().permute(1, 2, 0).numpy()

                psnr_val = psnr(real, fake, data_range=1.0)
                ssim_val = ssim(real, fake, data_range=1.0, multichannel=True, win_size=3)
                lpips_val = lpips_fn(
                    torch.tensor(downscaled_images[j]).unsqueeze(0).to(device).float() * 2 - 1,  # Convert to float32 and normalize to [-1, 1]
                    fake_images[j].unsqueeze(0) * 2 - 1,
                ).item()

                total_psnr += psnr_val
                total_ssim += ssim_val
                total_lpips += lpips_val
                count += 1

                save_image(fake_images[j], os.path.join(image_dir, f"validation_downscale_{scale_factor}_{i}_{j}_generated.png"))
                save_image(torch.tensor(downscaled_images[j]).to(device), os.path.join(image_dir, f"validation_downscale_{scale_factor}_{i}_{j}_downscaled.png"))

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count

    print(f"Validation Results with Downscaling (scale_factor={scale_factor}): PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}, LPIPS = {avg_lpips:.4f}")





# # 验证函数
# def validate(generator, loader, input_dim, image_dir, lpips_fn):
#     generator.eval()
#     total_psnr, total_ssim, total_lpips = 0, 0, 0
#     count = 0

#     with torch.no_grad():
#         for i, (real_images, _) in enumerate(tqdm(loader, desc="Validating")):
#             real_images = real_images.to(device)

#             # 生成噪声并生成图像
#             noise = torch.randn(real_images.size(0), input_dim).to(device)
#             fake_images = generator(noise)

#             for j in range(real_images.size(0)):
#                 real = real_images[j].cpu().permute(1, 2, 0).numpy()
#                 fake = fake_images[j].cpu().permute(1, 2, 0).numpy()

#                 # 计算 PSNR 和 SSIM
#                 psnr_val = psnr(real, fake, data_range=1.0)
#                 ssim_val = ssim(real, fake, data_range=1.0, multichannel=True, win_size=3)
#                 lpips_val = lpips_fn(
#                     real_images[j].unsqueeze(0) * 2 - 1,  # LPIPS input should be normalized to [-1, 1]
#                     fake_images[j].unsqueeze(0) * 2 - 1,
#                 ).item()

#                 total_psnr += psnr_val
#                 total_ssim += ssim_val
#                 total_lpips += lpips_val
#                 count += 1

#                 # 保存恢复的图像
#                 save_image(fake_images[j], os.path.join(image_dir, f"validation_{i}_{j}_generated.png"))
#                 save_image(real_images[j], os.path.join(image_dir, f"validation_{i}_{j}_real.png"))

#     avg_psnr = total_psnr / count
#     avg_ssim = total_ssim / count
#     avg_lpips = total_lpips / count

#     print(f"Validation Results: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}, LPIPS = {avg_lpips:.4f}")

def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    val_dataset = DIV2KDataset("./dataset/DIV2K_valid_LR_x8", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    input_dim = 100
    generator = Generator(input_dim).to(device)

    checkpoint_path = "./training_checkpoints_25/ckpt_epoch_0020.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    image_dir = "./validation_results"
    os.makedirs(image_dir, exist_ok=True)

    noise_levels = [0]

    validate_with_noise(generator, val_loader, input_dim, image_dir, lpips_fn, noise_levels)

if __name__ == "__main__":
    main()
