import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import numpy as np
from tqdm import tqdm
import sys
import random
from collections import Counter
from LightCNN.light_cnn import LightCNN_29Layers_v2

# Enable better CUDA error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Parse arguments
parser = argparse.ArgumentParser(description="Train LightCNN on face dataset")
parser.add_argument("--data-dir", type=str, default="data_processed/tinyfaces", 
                    help="Directory with class folders")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--cpu", action="store_true", help="Force CPU training")
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
parser.add_argument("--output", type=str, default="lightcnn_model.pth", 
                    help="Output model path")
args = parser.parse_args()

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class SafeFaceDataset(Dataset):
    """Dataset class with extensive validation and debug features"""
    
    def __init__(self, root_dir, transform=None, debug=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.debug = debug
        
        # Get all valid class directories
        class_dirs = sorted([d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d))])
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {root_dir}")
        
        print(f"Found {len(class_dirs)} potential class directories")
        
        # Create class mapping and store class details
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_counts = {}
        
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir] = idx
            self.idx_to_class[idx] = class_dir
            self.class_counts[class_dir] = 0
            
            # Print mapping in color
            print(f"\033[94mClass {class_dir} -> Index {idx}\033[0m")
        
        # Collect all valid images with extra checks
        print("Loading dataset images...")
        for class_dir in tqdm(class_dirs):
            class_path = os.path.join(root_dir, class_dir)
            class_idx = self.class_to_idx[class_dir]
            
            valid_images = 0
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('class_info.txt')):
                    continue  # Skip info files
                    
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    
                    # Basic validation checks
                    if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                        if self.debug:
                            print(f"Skipping empty file: {img_path}")
                        continue
                    
                    # Try opening the image to verify it's valid
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            if width <= 0 or height <= 0:
                                if self.debug:
                                    print(f"Skipping invalid image dimensions: {img_path}")
                                continue
                                
                        # Image is valid, add to dataset
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
                        self.class_counts[class_dir] += 1
                        valid_images += 1
                    except Exception as e:
                        if self.debug:
                            print(f"Error validating {img_path}: {e}")
                        continue
            
            # Print class statistics
            print(f"  Added {valid_images} images for class {class_dir} (index {class_idx})")
        
        # Verify label range
        if len(self.labels) > 0:
            label_min = min(self.labels)
            label_max = max(self.labels)
            print(f"Label range: {label_min} to {label_max} (expected range: 0 to {len(self.class_to_idx)-1})")
            
            if label_max >= len(self.class_to_idx) or label_min < 0:
                raise ValueError(f"Invalid label range: min={label_min}, max={label_max}, classes={len(self.class_to_idx)}")
            
            # Count labels
            label_counts = Counter(self.labels)
            print(f"Label distribution:")
            for label, count in sorted(label_counts.items()):
                class_name = self.idx_to_class.get(label, "UNKNOWN")
                print(f"  Label {label} ({class_name}): {count} images")
        
        print(f"Successfully loaded {len(self.image_paths)} images across {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Extra validation in debug mode
            if self.debug and (label < 0 or label >= len(self.class_to_idx)):
                print(f"⚠️ Invalid label {label} for image {img_path}")
                # Use a safe default
                label = 0
            
            # Load and process the image
            img = Image.open(img_path).convert('L')  # Grayscale
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return a fallback sample to avoid crashes
            dummy = torch.zeros(1, 128, 128)
            return dummy, 0

def train(args):
    """Main training function with improved error handling"""
    print(f"\n{'='*30} LightCNN Training {'='*30}")
    print(f"Dataset: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Debug mode: {args.debug}")
    print(f"{'='*75}\n")
    
    # Setup device
    if args.cpu:
        device = torch.device("cpu")
        print("Forcing CPU training")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Load dataset with thorough validation
    dataset = SafeFaceDataset(
        root_dir=args.data_dir,
        transform=transform,
        debug=args.debug
    )
    
    # Early exit if empty dataset
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
    
    # Create data loader with safety settings
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if args.debug else 2,  # Reduce workers in debug mode
        pin_memory=(device.type == "cuda"),
    )
    
    # Get number of classes from dataset
    num_classes = len(dataset.class_to_idx)
    print(f"Setting up model with {num_classes} output classes")
    
    # Create model
    model = LightCNN_29Layers_v2(num_classes=num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Print model summary
    print(f"Model: LightCNN_29Layers_v2")
    print(f"Input shape: [N, 1, 128, 128]")
    print(f"Output shape: [N, {num_classes}]")
    
    # Verify a batch before training
    print("\nVerifying first batch...")
    try:
        first_batch = next(iter(loader))
        images, labels = first_batch
        
        print(f"First batch shape: images={images.shape}, labels={labels.shape}")
        print(f"Label range: min={labels.min().item()}, max={labels.max().item()}")
        
        # Check if any labels are out of range
        if labels.max() >= num_classes or labels.min() < 0:
            print(f"⚠️ First batch contains invalid labels!")
            invalid_count = ((labels >= num_classes) | (labels < 0)).sum().item()
            print(f"Found {invalid_count} invalid labels in first batch")
            
            # Correct invalid labels for testing
            labels = torch.clamp(labels, 0, num_classes-1)
        
        # Try forward pass on CPU first
        with torch.no_grad():
            cpu_model = LightCNN_29Layers_v2(num_classes=num_classes)
            cpu_outputs = cpu_model(images)
            
            if isinstance(cpu_outputs, tuple):
                cpu_logits = cpu_outputs[1]
            else:
                cpu_logits = cpu_outputs
                
            print(f"CPU forward pass output shape: {cpu_logits.shape}")
            print("CPU validation succeeded!")
        
    except Exception as e:
        print(f"❌ Error during batch verification: {e}")
        print("There may be problems during training. Proceeding with caution.")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        # Training loop with extensive error handling
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            try:
                # Verify labels before sending to device
                if torch.any((labels < 0) | (labels >= num_classes)):
                    bad_indices = torch.where((labels < 0) | (labels >= num_classes))[0]
                    print(f"\n⚠️ Batch {i} contains {len(bad_indices)} invalid labels.")
                    print(f"Invalid labels: {labels[bad_indices].tolist()}")
                    print(f"Clamping labels to valid range...")
                    labels = torch.clamp(labels, 0, num_classes-1)
                
                # Move data to device
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                
                # Handle model output format
                if isinstance(outputs, tuple):
                    features, logits = outputs
                else:
                    logits = outputs
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                batch_count += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*correct/total:.1f}%"
                })
                
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    print(f"\n❌ CUDA error in batch {i}: {str(e)}")
                    print("This is likely due to invalid labels. Skipping batch.")
                    
                    # Try to recover by clearing GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    continue
                else:
                    print(f"\n❌ Runtime error in batch {i}: {str(e)}")
                    if args.debug:
                        raise
                    continue
                    
            except Exception as e:
                print(f"\n❌ Error in batch {i}: {str(e)}")
                if args.debug:
                    raise
                continue
        
        # Print epoch summary
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            accuracy = 100. * correct / total if total > 0 else 0
            print(f"\nEpoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        else:
            print(f"\nEpoch {epoch+1}/{args.epochs}: No valid batches processed")
    
    # Save model
    print(f"\nSaving model to {args.output}")
    torch.save({
        'state_dict': model.state_dict(),
        'class_to_idx': dataset.class_to_idx,
        'idx_to_class': dataset.idx_to_class
    }, args.output)
    print("Training complete!")

if __name__ == "__main__":
    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        if args.debug:
            raise