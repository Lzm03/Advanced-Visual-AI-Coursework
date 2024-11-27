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


def ssim_fn(pred, target):
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
    target = target.cpu().numpy().transpose(0, 2, 3, 1)

    total_ssim = 0
    for p, t in zip(pred, target):
        p = np.clip(p, 0, 1)
        t = np.clip(t, 0, 1)
        total_ssim += ssim(p, t, multichannel=True)

    return total_ssim / len(pred)


# 评估DIP模型
def evaluate_dip_model(model, valid_loader):
    model.eval()
    psnr_total = 0
    ssim_total = 0
    with torch.no_grad():
        for lr_images, hr_images in valid_loader:
            # lr_images, hr_images = lr_images.cuda(), hr_images.cuda()
            output = model(lr_images)

            psnr_total += psnr(output, hr_images)
            ssim_total += ssim_fn(output, hr_images)  # 使用修改后的 ssim_fn

    avg_psnr = psnr_total / len(valid_loader)
    avg_ssim = ssim_total / len(valid_loader)
    print(f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")


# 定义DIP模型
class DIPModel(nn.Module):
    def __init__(self):
        super(DIPModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)  # 不进行上采样

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.upsample(x)  # 不进行上采样，保持输入尺寸
        return x


# 数据集加载
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


# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 加载数据集
train_dataset = SuperResolutionDataset('./dataset/DIV2K_train_LR_x8/', './dataset/DIV2K_train_HR/', transform=transform)
valid_dataset = SuperResolutionDataset('./dataset/DIV2K_valid_LR_x8/', './dataset/DIV2K_valid_HR/', transform=transform)


# 随机抽取训练集和验证集的子集
def get_subset(dataset, subset_size=1000):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)


# 抽取20个样本的子集
train_subset = get_subset(train_dataset, subset_size=800)
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)


# 定义损失函数
def loss_fn(pred, target):
    return nn.MSELoss()(pred, target)


# 训练DIP模型
def train_dip_model(model, optimizer, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for lr_images, hr_images in train_loader:
            # lr_images, hr_images = lr_images.cuda(), hr_images.cuda()

            optimizer.zero_grad()
            output = model(lr_images)
            loss = loss_fn(output, hr_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    return model


# 可视化结果
def visualize_results(model, valid_loader):
    model.eval()
    with torch.no_grad():
        lr_images, hr_images = next(iter(valid_loader))
        # lr_images, hr_images = lr_images.cuda(), hr_images.cuda()
        output = model(lr_images)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(lr_images[0].cpu().permute(1, 2, 0).numpy())
        axs[0].set_title('Low Resolution')
        axs[1].imshow(output[0].cpu().permute(1, 2, 0).numpy())
        axs[1].set_title('DIP Model Output')
        axs[2].imshow(hr_images[0].cpu().permute(1, 2, 0).numpy())
        axs[2].set_title('High Resolution')
        plt.show()


# PSNR计算
def psnr(pred, target):
    mse = ((pred - target) ** 2).mean()
    return 10 * torch.log10(1.0 / mse)


# 初始化模型和优化器
def initialize_dip_model():
    model = DIPModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer


# 训练和评估
if __name__ == '__main__':
    model, optimizer = initialize_dip_model()
    trained_model = train_dip_model(model, optimizer, train_loader, num_epochs=1)
    evaluate_dip_model(trained_model, valid_loader)
    visualize_results(trained_model, valid_loader)
