import os
import cv2

def downscale_images(input_dir, output_dir, scale_factor=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, image_name)
        
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {image_name}")
            continue
        
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read image: {image_name}, skipping...")
            continue
        

        new_width = img.shape[1] // scale_factor
        new_height = img.shape[0] // scale_factor
        
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, resized_img)
        print(f"Saved resized image: {output_path}")

if __name__ == "__main__":
    input_dir = "../dataset/DIV2K_valid_LR_x8"
    output_dir = "./DIV2K_valid_LR_x16"
    
    downscale_images(input_dir, output_dir, scale_factor=2)
