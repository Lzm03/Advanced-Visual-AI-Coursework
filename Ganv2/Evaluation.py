import os
import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='alex').to(device)

def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=img2.max() - img2.min())

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, data_range=img2.max() - img2.min(),win_size=3)

def calculate_lpips(img1, img2):
    img1_tensor = torch.tensor(img1.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    img2_tensor = torch.tensor(img2.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    return lpips_model(img1_tensor, img2_tensor).item()

def resize_image(img, target_shape):
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

def evaluate_images(generated_dir, high_res_dir):
    results = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    
    generated_images = sorted(os.listdir(generated_dir))
    high_res_images = sorted(os.listdir(high_res_dir))
    
    if len(generated_images) != len(high_res_images):
        print("Mismatch in the number of images between the two directories.")
        return results, 0, 0, 0
    
    for gen_name, ref_name in zip(generated_images, high_res_images):
        gen_path = os.path.join(generated_dir, gen_name)
        ref_path = os.path.join(high_res_dir, ref_name)
        
        gen_img = cv2.imread(gen_path)
        ref_img = cv2.imread(ref_path)
        
        if gen_img is None or ref_img is None:
            print(f"Error reading {gen_name} or {ref_name}, skipping...")
            continue
        
        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        
        gen_img = resize_image(gen_img, ref_img.shape)
        
        psnr_value = calculate_psnr(gen_img, ref_img)
        ssim_value = calculate_ssim(gen_img, ref_img)
        lpips_value = calculate_lpips(gen_img, ref_img)
        
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
        
        results.append({
            "Generated Image": gen_name,
            "Reference Image": ref_name,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "LPIPS": lpips_value
        })
        
        print(f"{gen_name} vs {ref_name} - PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")
    
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0
    avg_lpips = np.mean(lpips_values) if lpips_values else 0
    
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
    
    return results, avg_psnr, avg_ssim, avg_lpips



if __name__ == "__main__":
    generated_dir = "./output_x16_HR"
    high_res_dir = "../dataset/DIV2K_valid_HR"
    
    results, avg_psnr, avg_ssim, avg_lpips = evaluate_images(generated_dir, high_res_dir)
    
    for result in results:
        print(result)
    
    print(f"Final Averages - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
