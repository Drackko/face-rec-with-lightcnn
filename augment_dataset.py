import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import random

def get_identity_counts(dataset_path):
    """
    Count images per identity in the dataset
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary of identity names with their image counts
    """
    identity_counts = {}
    
    # Assuming the directory structure is dataset_path/identity/images
    for identity in os.listdir(dataset_path):
        identity_dir = os.path.join(dataset_path, identity)
        
        # Skip if not a directory
        if not os.path.isdir(identity_dir):
            continue
        
        # Count image files
        image_files = [f for f in os.listdir(identity_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        identity_counts[identity] = len(image_files)
    
    return identity_counts

def apply_augmentations(image, rotation_range=(-10, 10), skew_range=(-0.1, 0.1)):
    """
    Apply rotation and skewing to image
    
    Args:
        image: Input image
        rotation_range: Tuple of (min_angle, max_angle) in degrees
        skew_range: Tuple of (min_skew, max_skew) factor
        
    Returns:
        Augmented image
    """
    height, width = image.shape[:2]
    
    # Random rotation
    angle = random.uniform(*rotation_range)
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                           borderMode=cv2.BORDER_REFLECT)
    
    # Random skewing (affine transformation)
    skew_factor = random.uniform(*skew_range)
    pts1 = np.float32([[0, 0], [width, 0], [0, height]])
    pts2 = np.float32([[0, 0], [width, 0], [int(skew_factor * width), height]])
    skew_matrix = cv2.getAffineTransform(pts1, pts2)
    skewed = cv2.warpAffine(rotated, skew_matrix, (width, height), 
                          borderMode=cv2.BORDER_REFLECT)
    
    return skewed

def augment_dataset(input_path, output_path, target_count=None, max_augmentations=5):
    """
    Augment dataset by adding rotated and skewed versions of images for identities with fewer images
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save augmented dataset
        target_count: Target number of images per identity (None = use median)
        max_augmentations: Maximum number of augmentations per original image
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get counts of images per identity
    print("Analyzing dataset...")
    identity_counts = get_identity_counts(input_path)
    
    if not identity_counts:
        print(f"No valid identities found in {input_path}")
        return
    
    # If target_count not specified, use median count
    if target_count is None:
        counts = list(identity_counts.values())
        target_count = int(np.median(counts))
        print(f"Setting target count to median: {target_count} images per identity")
    
    # Sort identities by count for reporting
    sorted_identities = sorted(identity_counts.items(), key=lambda x: x[1])
    
    # Prepare for tracking augmentation results
    results = {
        'Identity': [],
        'Original_Count': [],
        'Augmented_Added': [],
        'Final_Count': []
    }
    
    # Process each identity
    for identity, count in tqdm(sorted_identities, desc="Augmenting identities"):
        results['Identity'].append(identity)
        results['Original_Count'].append(count)
        
        src_dir = os.path.join(input_path, identity)
        dst_dir = os.path.join(output_path, identity)
        os.makedirs(dst_dir, exist_ok=True)
        
        # Copy all original images first
        augmented_count = 0
        image_files = [f for f in os.listdir(src_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in image_files:
            src_path = os.path.join(src_dir, img_file)
            dst_path = os.path.join(dst_dir, img_file)
            
            # Copy original image
            if not os.path.exists(dst_path):
                cv2.imwrite(dst_path, cv2.imread(src_path))
        
        # If below target count, augment
        if count < target_count:
            needed = target_count - count
            # Calculate augmentations per image, don't exceed max_augmentations
            augs_per_image = min(max_augmentations, (needed + count - 1) // count)
            
            # Create augmented versions
            for img_file in image_files:
                src_path = os.path.join(src_dir, img_file)
                img = cv2.imread(src_path)
                
                if img is None:
                    continue
                
                base_name = os.path.splitext(img_file)[0]
                
                # Create multiple augmentations of this image
                for i in range(augs_per_image):
                    if augmented_count >= needed:
                        break
                        
                    augmented = apply_augmentations(img)
                    aug_filename = f"{base_name}_aug{i+1}.jpg"
                    aug_path = os.path.join(dst_dir, aug_filename)
                    
                    cv2.imwrite(aug_path, augmented)
                    augmented_count += 1
        
        results['Augmented_Added'].append(augmented_count)
        results['Final_Count'].append(count + augmented_count)
    
    # Create report dataframe
    df = pd.DataFrame(results)
    
    # Save CSV report
    report_path = os.path.join(output_path, 'augmentation_report.csv')
    df.to_csv(report_path, index=False)
    
    # Generate summary statistics
    total_original = df['Original_Count'].sum()
    total_augmented = df['Augmented_Added'].sum()
    total_final = df['Final_Count'].sum()
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    plt.bar(df['Identity'], df['Original_Count'], label='Original')
    plt.bar(df['Identity'], df['Augmented_Added'], bottom=df['Original_Count'], label='Augmented')
    plt.xlabel('Identity')
    plt.ylabel('Number of Images')
    plt.title('Dataset Augmentation Results')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'augmentation_results.png'))
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET AUGMENTATION SUMMARY")
    print("="*50)
    print(f"Total identities: {len(df)}")
    print(f"Original images: {total_original}")
    print(f"Augmented images added: {total_augmented}")
    print(f"Final dataset size: {total_final}")
    print(f"Min images per identity (before): {df['Original_Count'].min()}")
    print(f"Min images per identity (after): {df['Final_Count'].min()}")
    print(f"Max images per identity (before): {df['Original_Count'].max()}")
    print(f"Max images per identity (after): {df['Final_Count'].max()}")
    print(f"Target images per identity: {target_count}")
    print(f"Report saved to: {report_path}")
    print("="*50)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment face dataset by adding rotated and skewed images")
    parser.add_argument("--input", default="/mnt/data/project_backups/DATASETS/ML-IInd/processed/",
                        help="Path to input dataset directory")
    parser.add_argument("--output", default="augmented_dataset",
                        help="Path to output the augmented dataset")
    parser.add_argument("--target",default=50, type=int, 
                        help="Target number of images per identity (default: median of original counts)")
    parser.add_argument("--max_aug", type=int, default=5,
                        help="Maximum number of augmentations per original image")
    
    args = parser.parse_args()
    
    augment_dataset(args.input, args.output, args.target, args.max_aug)