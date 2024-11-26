import torch
import torch.nn as nn

def discriminator_loss(real_output, fake_output, device):
    # Define the Binary Cross Entropy loss function
    loss_func = nn.BCELoss()

    # Loss for real images
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
    # Loss for fake images
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))

    loss_D = real_loss + fake_loss
    return loss_D

# Generator loss function (with L1 loss)
def generator_loss(fake_output, clean_imgs, generated_imgs):
    adversarial_loss = -torch.mean(fake_output)
    l1_loss = torch.nn.functional.l1_loss(generated_imgs, clean_imgs)
    total_loss = adversarial_loss + 100 * l1_loss  # Reduced weighting of L1 losses
    return total_loss