import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

def process_folder(input_folder, output_folder, noise_type="gaussian", **noise_params):
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"Skipping invalid file: {filename}")
                continue

            if noise_type == "gaussian":
                noisy_image = add_gaussian_noise(image, **noise_params)
            elif noise_type == "salt_and_pepper":
                noisy_image = add_salt_and_pepper_noise(image, **noise_params)
            else:
                print(f"Unsupported noise type: {noise_type}")
                continue
            cv2.imwrite(output_path, noisy_image)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "./DIV2K_valid_LR_x8"
    output_folder = "./noise_images_5_LR_x8"
    noise_type = "gaussian"

    noise_params = {
        "mean": 0,
        "std": 5
    }


    process_folder(input_folder, output_folder, noise_type=noise_type, **noise_params)
