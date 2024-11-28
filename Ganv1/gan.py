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
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

device = torch.device("mps")
lpips_model = lpips.LPIPS(net='alex').to(device)
torch.backends.cudnn.benchmark = True

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)
        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)

        return lr_image, hr_image
    

transform_lr = transforms.Compose([
    transforms.Resize((32,32)), 
    transforms.ToTensor(), 
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_hr = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(), 
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_dataset = DIV2KDataset('./dataset/DIV2K_train_LR_x8', './dataset/DIV2K_train_HR',
                             transform_lr=transform_lr, transform_hr=transform_hr)
val_dataset = DIV2KDataset('./dataset/DIV2K_valid_LR_x8', './dataset/DIV2K_valid_HR',
                           transform_lr=transform_lr, transform_hr=transform_hr)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
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
        )
        self.fc = None 

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        if self.fc is None or x.size(1) != self.fc.in_features:
            in_features = x.size(1)
            self.fc = nn.Linear(in_features, 1).to(x.device)
        output = torch.sigmoid(self.fc(x))
        return output

loss_func = nn.BCELoss()

def generator_loss(fake_output, fake_image, real_image):
    adv_loss = nn.BCELoss()(fake_output, torch.ones_like(fake_output))
    pixel_loss = nn.MSELoss()(fake_image, real_image)
    return adv_loss + 0.01 * pixel_loss


def discriminator_loss(real_output, fake_output):
    real_loss = nn.BCELoss()(real_output, torch.ones_like(real_output))
    fake_loss = nn.BCELoss()(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss

def save_model(generator, discriminator, optim_G, optim_D, epoch, save_dir="./model_checkpoints_2"):

    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optim_G_state_dict': optim_G.state_dict(),
        'optim_D_state_dict': optim_D.state_dict()
    }
    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved for epoch {epoch} at {save_path}.")


def load_model(checkpoint_path, generator, discriminator, optim_G, optim_D):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'],strict=False)
    
    discriminator_state_dict = checkpoint['discriminator_state_dict']
    model_state_dict = discriminator.state_dict()
    updated_state_dict = {k: v for k, v in discriminator_state_dict.items() if k in model_state_dict}
    model_state_dict.update(updated_state_dict)
    discriminator.load_state_dict(model_state_dict)
    # optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
    # optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
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
                # hr_image_np = hr_images[i].permute(1, 2, 0).cpu().numpy()
                # fake_image_np = fake_images[i].permute(1, 2, 0).cpu().numpy()

                hr_image_np = ((hr_images[i].permute(1, 2, 0).cpu().numpy() + 1) / 2).astype('float32')
                fake_image_np = ((fake_images[i].permute(1, 2, 0).cpu().numpy() + 1) / 2).astype('float32')

                psnr_value = psnr(hr_image_np, fake_image_np, data_range=1.0)
                ssim_value = ssim(
                    hr_image_np,
                    fake_image_np,
                    data_range=1.0, 
                    win_size=None, 
                    channel_axis=-1,  
                )
                lpips_value = lpips_model(
                    hr_images[i].unsqueeze(0),
                    fake_images[i].unsqueeze(0)
                ).item()

                total_psnr += psnr_value
                total_ssim += ssim_value
                total_lpips += lpips_value
                total_samples += 1

    if total_samples > 0:
        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
        avg_lpips = total_lpips / total_samples
        print(f"Validation - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
        return avg_psnr, avg_ssim, avg_lpips
    else:
        print("Validation set is empty!")
        return 0.0, 0.0, 0.0

    
def save_images(epoch, lr_images, hr_images, fake_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (lr, hr, fake) in enumerate(zip(lr_images, hr_images, fake_images)):
        save_image(lr, os.path.join(save_dir, f"epoch_{epoch}_lr_{i}.png"))
        save_image(hr, os.path.join(save_dir, f"epoch_{epoch}_hr_{i}.png"))
        save_image(fake, os.path.join(save_dir, f"epoch_{epoch}_fake_{i}.png"))




generator = Generator().to(device)
discriminator = Discriminator().to(device)

optim_G = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_D = Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

# start_epoch = load_model("./model_checkpoints/checkpoint_epoch_25.pth", generator, discriminator, optim_G, optim_D)
# optim_G = Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
# optim_D = Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))


start_epoch = 0
results_dir = "./training_result_4"

from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler_G = ReduceLROnPlateau(optim_G, mode='max', patience=5, factor=0.5, verbose=True)
scheduler_D = ReduceLROnPlateau(optim_D, mode='max', patience=5, factor=0.5, verbose=True)

gen_update_steps = 1 
disc_update_steps = 1 

best_psnr = 0.0 

for epoch in range(start_epoch, 50):
    generator.train()
    discriminator.train()
    total_loss_D, total_loss_G = 0.0, 0.0

    for batch_idx, (lr_images, hr_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        for _ in range(disc_update_steps):
            fake_images = generator(lr_images)
            real_output = discriminator(hr_images)
            fake_output = discriminator(fake_images.detach())
            loss_D = discriminator_loss(real_output, fake_output)

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        for _ in range(gen_update_steps):
            fake_images = generator(lr_images)
            fake_output = discriminator(fake_images)
            loss_G = generator_loss(fake_output, fake_images, hr_images)

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

        if batch_idx == 0:
            save_dir = os.path.join(results_dir, f"epoch_{epoch + 1}")
            save_images(epoch + 1, lr_images.cpu(), hr_images.cpu(), fake_images.cpu(), save_dir)

    print(f"Epoch {epoch + 1}, Loss_D: {total_loss_D / len(train_loader):.4f}, Loss_G: {total_loss_G / len(train_loader):.4f}")


    avg_psnr, avg_ssim, avg_lpips = validate(generator, val_loader)
    scheduler_G.step(avg_psnr)
    scheduler_D.step(avg_psnr)

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        disc_update_steps = min(disc_update_steps + 1, 3)
        print(f"PSNR improved to {avg_psnr:.4f}. Increasing discriminator update steps to {disc_update_steps}.")
    else:
        gen_update_steps = min(gen_update_steps + 1, 3)
        print(f"PSNR did not improve. Increasing generator update steps to {gen_update_steps}.")

    save_model(generator, discriminator, optim_G, optim_D, epoch=epoch + 1)


