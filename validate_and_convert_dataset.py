import os
import shutil
import random
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

def validate_and_convert_dataset(input_dir, output_dir, min_images_per_class=5, max_images_per_class=200):
    """
    Validates each image in the dataset and creates a clean, validated copy
    with a simplified structure.
    """
    print(f"Validating and converting dataset from {input_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track statistics
    total_images = 0
    valid_images = 0
    skipped_images = 0
    valid_classes = 0
    
    # Get all subdirectories (person classes)
    person_dirs = [d for d in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(person_dirs)} potential class directories")
    
    # Process each person directory
    for idx, person in enumerate(person_dirs):
        person_input_dir = os.path.join(input_dir, person)
        
        # Validate and collect images for this person
        valid_person_images = []
        
        # Get all files in this directory
        all_files = os.listdir(person_input_dir)
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Processing directory {person} ({idx+1}/{len(person_dirs)}) - Found {len(image_files)} potential images")
        
        # Validate each image
        for img_name in tqdm(image_files, desc=f"Validating {person}", leave=False):
            img_path = os.path.join(person_input_dir, img_name)
            total_images += 1
            
            try:
                # Check if file exists and is not empty
                if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                    print(f"  ⚠️ Skipping empty or non-existent file: {img_path}")
                    skipped_images += 1
                    continue
                
                # Try to open the image
                with Image.open(img_path) as img:
                    # Convert to grayscale to check format
                    img_gray = img.convert('L')
                    
                    # Check dimensions
                    width, height = img_gray.size
                    if width <= 10 or height <= 10:
                        print(f"  ⚠️ Skipping too small image: {img_path} ({width}x{height})")
                        skipped_images += 1
                        continue
                    
                    # Verify pixel values
                    img_array = np.array(img_gray)
                    if img_array.min() == img_array.max():
                        print(f"  ⚠️ Skipping uniform image: {img_path} (all pixels same value)")
                        skipped_images += 1
                        continue
                
                # Image is valid
                valid_person_images.append(img_path)
                valid_images += 1
                
            except Exception as e:
                print(f"  ⚠️ Error validating {img_path}: {e}")
                skipped_images += 1
        
        # Check if we have enough valid images for this class
        if len(valid_person_images) < min_images_per_class:
            print(f"  ⚠️ Skipping class {person} with only {len(valid_person_images)} valid images (minimum {min_images_per_class})")
            continue
        
        # Create output directory for this person
        person_output_dir = os.path.join(output_dir, f"class_{idx:04d}")
        os.makedirs(person_output_dir, exist_ok=True)
        
        # Create a mapping file to remember original class name
        with open(os.path.join(person_output_dir, "class_info.txt"), "w") as f:
            f.write(f"Original directory: {person}\n")
            f.write(f"Valid images: {len(valid_person_images)}\n")
        
        # Limit maximum images per class if needed
        if max_images_per_class and len(valid_person_images) > max_images_per_class:
            print(f"  ℹ️ Limiting class {person} from {len(valid_person_images)} to {max_images_per_class} images")
            valid_person_images = random.sample(valid_person_images, max_images_per_class)
        
        # Copy valid images to output directory
        for i, img_path in enumerate(valid_person_images):
            img_ext = os.path.splitext(img_path)[1]
            dest_path = os.path.join(person_output_dir, f"image_{i:04d}{img_ext}")
            
            try:
                # Open, convert to grayscale, resize, and save
                with Image.open(img_path) as img:
                    img_gray = img.convert('L')
                    img_resized = img_gray.resize((128, 128), Image.LANCZOS)
                    img_resized.save(dest_path)
            except Exception as e:
                print(f"  ⚠️ Error copying {img_path}: {e}")
                continue
        
        valid_classes += 1
        print(f"  ✅ Processed class {person}: {len(valid_person_images)} images -> {person_output_dir}")
    
    # Print summary
    print("\n=== Dataset Conversion Summary ===")
    print(f"Total images scanned: {total_images}")
    print(f"Valid images: {valid_images} ({valid_images/total_images*100:.1f}%)")
    print(f"Skipped images: {skipped_images} ({skipped_images/total_images*100:.1f}%)")
    print(f"Valid classes: {valid_classes} out of {len(person_dirs)}")
    print(f"Clean dataset saved to: {output_dir}")
    
    # Create a class mapping file
    with open(os.path.join(output_dir, "class_mapping.txt"), "w") as f:
        f.write(f"Total classes: {valid_classes}\n")
        f.write("Format: class_dir_name -> original_name\n\n")
        
        for idx, person in enumerate(person_dirs):
            class_dir = os.path.join(output_dir, f"class_{idx:04d}")
            if os.path.exists(class_dir):
                f.write(f"class_{idx:04d} -> {person}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and convert face dataset")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing class folders with images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for clean dataset")
    parser.add_argument("--min-per-class", type=int, default=5, help="Minimum images per class")
    parser.add_argument("--max-per-class", type=int, default=200, help="Maximum images per class")
    
    args = parser.parse_args()
    validate_and_convert_dataset(args.input, args.output, args.min_per_class, args.max_per_class)