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

class DIV2KDataset(Dataset):
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


transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    # transforms.RandomHorizontalFlip(p=0.5), 
    # transforms.RandomRotation(15),         
    transforms.ToTensor()             
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])



train_dataset = DIV2KDataset(root_dir='./dataset/DIV2K_train_LR_x8', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_dataset = DIV2KDataset(root_dir='./dataset/DIV2K_valid_LR_x8', transform=val_transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256 * 8 * 8)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.br1(self.fc1(x))
        x = x.view(-1, 256, 8, 8)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        output = self.conv4(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        output = self.fc1(x)
        return output

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
    for _ in range(2):
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
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_samples = 0

    with torch.no_grad():
        for real_images, _ in dataloader:
            real_images = real_images.to(device) 
            noise = torch.randn(real_images.size(0), input_dim).to(device)
            fake_images = generator(noise)  

      
            for i in range(real_images.size(0)):
                psnr_value = psnr(
                    real_images[i].cpu().numpy(),
                    fake_images[i].cpu().numpy(),
                    data_range=1.0
                )
                total_psnr += psnr_value

      
            for i in range(real_images.size(0)):
                real_image_np = real_images[i].permute(1, 2, 0).cpu().numpy()
                fake_image_np = fake_images[i].permute(1, 2, 0).cpu().numpy()
                ssim_value = ssim(
                    real_image_np,
                    fake_image_np,
                    multichannel=True,
                    data_range=1.0,
                    win_size=3
                )
                total_ssim += ssim_value

            lpips_value = lpips_model(
                real_images,
                fake_images
            ).mean().item()
            total_lpips += lpips_value

            total_samples += real_images.size(0)


    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_lpips = total_lpips / total_samples


    print(f"Validation - Epoch {epoch + 1}:")
    print(f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")


    with open(os.path.join(image_dir, "validation_metrics.txt"), "a") as f:
        f.write(f"Epoch {epoch + 1}: PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}\n")

    generator.train()


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
    
# def load_checkpoint(checkpoint_path, generator, discriminator, optim_G, optim_D):
#     checkpoint = torch.load(checkpoint_path)
#     # generator.load_state_dict(checkpoint['generator_state_dict'])
#     # discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
#     # optim_G.load_state_dict(checkpoint['optim_G_state_dict'])
#     # optim_D.load_state_dict(checkpoint['optim_D_state_dict'])
#     start_epoch = checkpoint['epoch']
#     print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
#     return start_epoch


    
input_dim = 100
batch_size = 64
num_epoch = 50
lr = 0.0001
checkpoint_dir = './training_checkpoints_28'
os.makedirs(checkpoint_dir, exist_ok=True)
image_dir = './generated_image_28/'
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
checkpoint_path = './training_checkpoints_27/ckpt_epoch_0032.pth'

start_epoch = 0
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(checkpoint_path, gan_G, gan_D, optim_G, optim_D)
else:
    print("No checkpoint found. Starting training from scratch.")

# if os.path.exists(checkpoint_path):
#     start_epoch = load_checkpoint(checkpoint_path, gan_G, gan_D, optim_G, optim_D)
# else:
#     print("No checkpoint found. Starting training from scratch.")
#     start_epoch = 0

# Initialise the list to store the losses for each epoch
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


# def visualise_generated_images(generator, input_dim, epoch, image_dir, num_images=64):
#     """
#     Parameters: 
#         - num_images: the number of images generated, defaults to 64.
#     """ 
#     generator.eval()
#     with torch.no_grad():
#         x = torch.randn(num_images, input_dim).to(device)
#         img = generator(x)
#     save_image(img, f'{image_dir}/generated_images/{epoch + 1}_epoch.png') 
#     generator.train()

# def display_image(epoch_no):
#     return Image.open(f'{image_dir}{epoch_no}_epoch.png')

# def save_checkpoint(epoch, generator, discriminator, optim_G, optim_D, checkpoint_dir):
#     """
#     Save the model and optimizer states at a specific epoch.
#     """
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
#     torch.save({
#         'epoch': epoch,
#         'generator_state_dict': generator.state_dict(),
#         'discriminator_state_dict': discriminator.state_dict(),
#         'optim_G_state_dict': optim_G.state_dict(),
#         'optim_D_state_dict': optim_D.state_dict(),
#     }, checkpoint_path)
#     print(f"Checkpoint saved at {checkpoint_path}")
    
# def log_metrics(summary_writer, step, epoch, loss_D, loss_G, data_load_time, step_time):
#     """
#     Logs metrics to TensorBoard.
    
#     Args:
#         summary_writer (SummaryWriter): TensorBoard summary writer instance.
#         step (int): Current training step.
#         epoch (int): Current epoch.
#         loss_D (float): Discriminator loss.
#         loss_G (float): Generator loss.
#         data_load_time (float): Time taken to load data in the current step.
#         step_time (float): Total time taken for the current step.
#     """
#     summary_writer.add_scalar("Epoch", epoch, step)
#     summary_writer.add_scalar("Loss/Discriminator", loss_D, step)
#     summary_writer.add_scalar("Loss/Generator", loss_G, step)
#     summary_writer.add_scalar("Time/DataLoad", data_load_time, step)
#     summary_writer.add_scalar("Time/Step", step_time, step)



# def validate(generator, dataloader, epoch, summary_writer):
#     generator.eval()
#     total_psnr = 0
#     total_mse = 0
#     with torch.no_grad():
#         for real_images, _ in dataloader:
#             real_images = real_images.to(device)
#             noise = torch.randn(real_images.size(0), args.input_dim).to(device)
#             fake_images = generator(noise)
#             mse = F.mse_loss(fake_images, real_images)
#             total_mse += mse.item()
#             total_psnr += psnr(real_images.cpu().numpy(), fake_images.cpu().numpy())

#     avg_psnr = total_psnr / len(dataloader)
#     avg_mse = total_mse / len(dataloader)
#     summary_writer.add_scalar("Validation/PSNR", avg_psnr, epoch)
#     summary_writer.add_scalar("Validation/MSE", avg_mse, epoch)
#     print(f"Validation - Epoch {epoch}: Average PSNR: {avg_psnr:.2f}, Average MSE: {avg_mse:.4f}")

# def train(args):
#     summary_writer = SummaryWriter(log_dir=args.log_dir)
#     os.makedirs(args.image_dir, exist_ok=True)
#     os.makedirs(args.checkpoint_dir, exist_ok=True)

#     transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
#     train_dataset = DIV2KDataset(root_dir="./dataset/DIV2K_train_LR_x8", transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_count)

#     valid_dataset = DIV2KDataset(root_dir="./dataset/DIV2K_valid_LR_x8", transform=transform)
#     valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker_count)

#     gan_G = Generator(args.input_dim).to(device)
#     gan_D = Discriminator().to(device)
#     optim_G = Adam(gan_G.parameters(), lr=args.lr)
#     optim_D = Adam(gan_D.parameters(), lr=args.lr)

#     step = 0 
#     for epoch in range(args.epochs):
#         gan_G.train()
#         gan_D.train()
#         total_loss_D, total_loss_G = 0, 0

#         for i, (real_images, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
#             real_images = real_images.to(device)
#             real_output = gan_D(real_images)
#             fake_images = gan_G(torch.randn([args.batch_size, args.input_dim]).to(device)).detach()
#             fake_output = gan_D(fake_images)
#             loss_D = discriminator_loss(real_output, fake_output)

#             # Train Discriminator
#             optim_D.zero_grad()
#             loss_D.backward()
#             optim_D.step()

#             # Train Generator
#             fake_images = gan_G(torch.randn([args.batch_size, args.input_dim]).to(device))
#             fake_output = gan_D(fake_images)
#             loss_G = generator_loss(fake_output)
            
#             optim_G.zero_grad()
#             loss_G.backward()
#             optim_G.step()

#             total_loss_D += loss_D.item()
#             total_loss_G += loss_G.item()

#         # Save generated and original images every epoch
#         visualise_generated_images(gan_G, args.input_dim, epoch + 1, real_images, args.image_dir)

#         print(f"Epoch {epoch + 1}/{args.epochs} | Loss_D: {total_loss_D / len(train_loader):.4f} | Loss_G: {total_loss_G / len(train_loader):.4f}")

#         if (epoch + 1) % 5 == 0:
#             validate(gan_G, valid_loader, epoch + 1, summary_writer)


#     summary_writer.close()

# parser = argparse.ArgumentParser(description="GAN Training Script")
# parser.add_argument("--batch-size", default=128, type=int, help="Batch size for training")
# parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
# parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate")
# parser.add_argument("--log-dir", default="logs", type=str, help="Directory for TensorBoard logs")
# parser.add_argument("--image-dir", default="generated_images", type=str, help="Directory for generated images")
# parser.add_argument("--checkpoint-dir", default="checkpoints", type=str, help="Directory for saving checkpoints")
# parser.add_argument("--input-dim", default=100, type=int, help="Input dimension for the generator")
# parser.add_argument("--worker-count", default=cpu_count(), type=int, help="Number of data loader workers")
# parser.add_argument("--log-frequency", default=5, type=int, help="Frequency of logging to TensorBoard")
# args = parser.parse_args()

# if __name__ == "__main__":
#     train(args)
