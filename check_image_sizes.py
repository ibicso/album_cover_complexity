import cv2
import numpy as np
import os

def check_image_sizes():
    print("CHECKING IMAGE SIZES:")
    print("=" * 50)
    
    # Check test images
    test_dir = 'test/test_images'
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:3]
        print('TEST IMAGES:')
        for f in test_files:
            img_path = os.path.join(test_dir, f)
            img = cv2.imread(img_path)
            if img is not None:
                pixels = img.shape[0] * img.shape[1] * img.shape[2]
                print(f'  {f}: {img.shape} = {pixels:,} pixels')
                print(f'    > 1e4 threshold: {pixels > 1e4}')
    else:
        print('TEST directory not found')

    print()
    
    # Check billboard/data images  
    data_dir = './data/img_all'
    if os.path.exists(data_dir):
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')][:3]
        print('BILLBOARD IMAGES:')
        for f in data_files:
            img_path = os.path.join(data_dir, f)
            img = cv2.imread(img_path)
            if img is not None:
                pixels = img.shape[0] * img.shape[1] * img.shape[2]
                print(f'  {f}: {img.shape} = {pixels:,} pixels')
                print(f'    > 1e4 threshold: {pixels > 1e4}')
    else:
        print('DATA directory not found')
    
    print()
    print("PATCH SIZE DIFFERENCES:")
    print("Small images (≤ 1e4 pixels):")
    for level in range(4):
        patch_size = 2 * (level + 1)
        print(f"  Level {level}: patch_size = {patch_size}")
    
    print("Large images (> 1e4 pixels):")
    for level in range(4):
        patch_size = 4 * (2 ** level)
        print(f"  Level {level}: patch_size = {patch_size}")

if __name__ == "__main__":
    check_image_sizes() 