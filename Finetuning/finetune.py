import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import time
import copy
import math
import warnings

# Import LightCNN model
from LightCNN.light_cnn import LightCNN_29Layers_v2

# Import EDSR from EDSR.py instead of defining it inline
from EDSR import EDSR, TFModelHandler, process_images

# TensorFlow availability flag
TF_AVAILABLE = False
print("TensorFlow is not available. Using PyTorch models only.")

# Custom dataset for TinyFaces with EDSR super-resolution
class TinyFacesDataset(Dataset):
    def __init__(self, root_dir, transform=None, sr_transform=None, apply_sr=False, 
                 sr_model=None, tf_model=None, min_size=60, target_size=128):
        self.root_dir = root_dir
        self.transform = transform
        self.sr_transform = sr_transform  # Transform for SR input
        self.apply_sr = apply_sr
        self.sr_model = sr_model  # PyTorch EDSR model
        self.tf_model = tf_model  # TensorFlow EDSR model
        self.min_size = min_size  # Min size threshold to apply SR
        self.target_size = target_size  # Final target size for the model
        self.images = []
        self.labels = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Assuming structure: root_dir/identity/images.jpg
        for identity_idx, identity in enumerate(os.listdir(root_dir)):
            identity_dir = os.path.join(root_dir, identity)
            if os.path.isdir(identity_dir):
                for img_name in os.listdir(identity_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(identity_idx)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')  # Use RGB
        
        # Determine if super-resolution is needed based on image size
        needs_sr = False
        if self.apply_sr and (self.sr_model is not None or self.tf_model is not None):
            width, height = image.size
            if width < self.min_size or height < self.min_size:
                needs_sr = True
                
        if needs_sr:
            # Convert grayscale to RGB by repeating channel 3 times
            rgb_image = Image.merge('RGB', [image, image, image])
            
            # Apply initial transform for SR input
            if self.sr_transform:
                sr_input = self.sr_transform(rgb_image)
            else:
                sr_input = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])(rgb_image)
            
            # Apply super-resolution
            with torch.no_grad():
                sr_input = sr_input.unsqueeze(0).to(self.device)
                sr_output = self.sr_model(sr_input)
                
                # Adjust SR output handling
                if sr_output.size(1) == 3:
                    # Convert RGB to grayscale using luminance formula
                    sr_output = 0.2989 * sr_output[:,0:1,:,:] + 0.5870 * sr_output[:,1:2,:,:] + 0.1140 * sr_output[:,2:3,:,:]
                
                # Convert back to PIL for further processing
                # Denormalize: from [-1,1] to [0,1] range
                sr_output = sr_output.squeeze(0).cpu()
                sr_output = (sr_output + 1) / 2
                sr_output = torch.clamp(sr_output, 0, 1)
                sr_image = transforms.ToPILImage()(sr_output)
                
                # Ensure output size is the target size
                if sr_image.size != (self.target_size, self.target_size):
                    sr_image = sr_image.resize((self.target_size, self.target_size), Image.BICUBIC)
                
                image = sr_image
        
        # Apply final transform for the model
        if self.transform:
            image = self.transform(image)
            
        return image, label, os.path.basename(os.path.dirname(img_path))  # Return image, label, and identity name

# Function to train EDSR model for super-resolution
def train_edsr(data_dir, output_path, scale=4, num_epochs=50, batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for EDSR training")
    
    # Initialize EDSR model from imported class
    edsr_model = EDSR(scale=scale, n_resblocks=16, n_feats=64, res_scale=1).to(device)
    
    # Create special dataset for SR training that returns both LR and HR images
    class SRDataset(Dataset):
        def __init__(self, image_paths, scale=4, lr_size=32, hr_size=128):
            self.image_paths = image_paths
            self.scale = scale
            self.lr_size = lr_size
            self.hr_size = hr_size
            
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            # Load image
            img = Image.open(img_path).convert('L')
            
            # Create HR image at target size
            hr_transform = transforms.Compose([
                transforms.Resize((self.hr_size, self.hr_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            hr_img = hr_transform(img)
            
            # Create LR image 
            lr_transform = transforms.Compose([
                transforms.Resize((self.lr_size, self.lr_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            lr_img = lr_transform(img)
            
            return lr_img, hr_img
    
    # Collect image paths - modified to not use 'train' subdirectory
    train_images = []
    for identity in os.listdir(data_dir):
        identity_dir = os.path.join(data_dir, identity)
        if os.path.isdir(identity_dir):
            for img_name in os.listdir(identity_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    train_images.append(os.path.join(identity_dir, img_name))
    
    # Use a subset of images for EDSR training to save time
    max_images = min(5000, len(train_images))  # Limit to 5000 images
    np.random.shuffle(train_images)
    train_images = train_images[:max_images]
    print(f"Using {len(train_images)} images for EDSR training")
    
    # Create datasets and dataloaders
    sr_dataset = SRDataset(train_images, scale=scale)
    sr_loader = DataLoader(sr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(edsr_model.parameters(), lr=1e-4)
    
    # Training loop
    best_loss = float('inf')
    best_model_weights = None
    
    print(f"Starting EDSR training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        edsr_model.train()
        running_loss = 0.0
        
        for lr_imgs, hr_imgs in sr_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            sr_imgs = edsr_model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(edsr_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * lr_imgs.size(0)
        
        epoch_loss = running_loss / len(sr_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = copy.deepcopy(edsr_model.state_dict())
    
    # Load best model weights
    edsr_model.load_state_dict(best_model_weights)
    
    # Save the trained model
    torch.save(edsr_model.state_dict(), output_path)
    print(f"EDSR model trained and saved to {output_path}")
    
    return edsr_model

# Function to finetune the LightCNN model with EDSR super-resolution
def finetune_lightcnn(model_path, data_dir, output_path, num_classes, 
                      sr_model_path=None, train_sr=False, num_epochs=10,
                      val_split=0.2):  # Added val_split parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Handle SR model loading - prefer .pb (TensorFlow) over .pth (PyTorch)
    sr_model = None
    tf_model = None
    
    if sr_model_path and os.path.exists(sr_model_path):
        # Skip TensorFlow models if TF is not available
        if sr_model_path.endswith('.pb') or os.path.isdir(sr_model_path):
            print(f"Cannot load TensorFlow model from {sr_model_path} (TensorFlow not available)")
            print("Will try to train a PyTorch model instead")
            sr_model = None
        else:
            # PyTorch model loading code remains the same
            try:
                print(f"Loading PyTorch EDSR model from {sr_model_path}")
                # Create verbose EDSR model configuration printout
                print(f"EDSR configuration: scale=4, n_colors=3")
                sr_model = EDSR(scale=4, n_colors=3).to(device)  # Use RGB (3 channels) to match pretrained weights
                
                # Print model structure
                print(f"EDSR model structure: {sr_model}")
                
                # Load and print checkpoint info
                checkpoint = torch.load(sr_model_path, map_location=device)
                print(f"Checkpoint keys: {checkpoint.keys()}")
                
                # Continue with existing code...
                new_state_dict = {}
                for k, v in checkpoint.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # Try loading with more error checking
                try:
                    sr_model.load_state_dict(new_state_dict)
                    print("EDSR model loaded successfully!")
                except Exception as load_error:
                    print(f"Failed to load state dict: {load_error}")
                    print("Keys in checkpoint vs model:")
                    model_keys = set(sr_model.state_dict().keys())
                    checkpoint_keys = set(new_state_dict.keys())
                    print(f"Keys in model but not checkpoint: {model_keys - checkpoint_keys}")
                    print(f"Keys in checkpoint but not model: {checkpoint_keys - model_keys}")
                    sr_model = None
            except Exception as e:
                print(f"Failed to create EDSR model: {e}")
                sr_model = None
    
    # Train SR model if needed
    if train_sr and sr_model is None and tf_model is None:
        print("Training EDSR model for super-resolution...")
        sr_output_path = sr_model_path if sr_model_path else "edsr_trained.pth"
        
        # Update train_edsr function call to not use 'train' subdirectory
        sr_model = train_edsr(data_dir, sr_output_path)
    
    # Set models to evaluation mode
    if sr_model is not None:
        sr_model.eval()
    
    # Check if we have a working SR model
    apply_sr = (sr_model is not None or tf_model is not None)
    if not apply_sr:
        print("EDSR model not available. Proceeding without super-resolution.")
    
    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # SR input transform (used when SR is applied)
    sr_input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create a list of all image paths and labels with validation
    all_images = []
    all_labels = []
    
    # Scan through all identity folders and collect images, ensuring valid labels
    valid_identities = []
    for identity_idx, identity in enumerate(os.listdir(data_dir)):
        identity_dir = os.path.join(data_dir, identity)
        if os.path.isdir(identity_dir):
            # Count images in this identity folder
            img_count = len([img for img in os.listdir(identity_dir) 
                           if img.endswith(('.jpg', '.jpeg', '.png'))])
            
            # Only include identities with at least one image
            if img_count > 0:
                valid_identities.append(identity)
    
    # Create a clean mapping from identities to contiguous indices
    identity_to_idx = {identity: idx for idx, identity in enumerate(valid_identities)}
    print(f"Using {len(identity_to_idx)} valid identities with contiguous indices")
    
    # Now collect images with validated labels
    for identity in valid_identities:
        identity_dir = os.path.join(data_dir, identity)
        for img_name in os.listdir(identity_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(identity_dir, img_name)
                all_images.append(img_path)
                all_labels.append(identity_to_idx[identity])
    
    # Verify all labels are in the valid range
    labels_array = np.array(all_labels)
    assert np.all(labels_array >= 0) and np.all(labels_array < len(identity_to_idx)), \
        "Invalid labels detected before training"
    print(f"Label range: {labels_array.min()} to {labels_array.max()}")
    
    # Shuffle the data and split into train/val sets
    indices = list(range(len(all_images)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(np.floor(val_split * len(indices)))
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]
    
    train_images = [all_images[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_images = [all_images[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"Train set: {len(train_images)} images, Val set: {len(val_images)} images")
    
    # Create custom datasets for train and val
    class SimpleDataset(Dataset):
        def __init__(self, images, labels, transform=None, apply_sr=False, 
                     sr_model=None, tf_model=None, sr_transform=None, min_size=60, target_size=128):
            self.images = images
            self.labels = labels
            self.transform = transform
            self.apply_sr = apply_sr
            self.sr_model = sr_model
            self.tf_model = tf_model
            self.sr_transform = sr_transform
            self.min_size = min_size
            self.target_size = target_size
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            img_path = self.images[idx]
            label = self.labels[idx]
            
            # Load image as grayscale
            image = Image.open(img_path).convert('L')  # Convert to grayscale (not RGB)
            
            # Determine if super-resolution is needed based on image size
            needs_sr = False
            if self.apply_sr and (self.sr_model is not None or self.tf_model is not None):
                width, height = image.size
                if width < self.min_size or height < self.min_size:
                    needs_sr = True
                    
            if needs_sr:
                # Convert grayscale to RGB by repeating channel 3 times
                rgb_image = Image.merge('RGB', [image, image, image])
                
                # Apply initial transform for SR input
                if self.sr_transform:
                    sr_input = self.sr_transform(rgb_image)
                else:
                    sr_input = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])(rgb_image)
                
                # Apply super-resolution
                with torch.no_grad():
                    sr_input = sr_input.unsqueeze(0).to(self.device)
                    sr_output = self.sr_model(sr_input)
                    
                    # Adjust SR output handling
                    if sr_output.size(1) == 3:
                        # Convert RGB to grayscale using luminance formula
                        sr_output = 0.2989 * sr_output[:,0:1,:,:] + 0.5870 * sr_output[:,1:2,:,:] + 0.1140 * sr_output[:,2:3,:,:]
                    
                    # Convert back to PIL for further processing
                    # Denormalize: from [-1,1] to [0,1] range
                    sr_output = sr_output.squeeze(0).cpu()
                    sr_output = (sr_output + 1) / 2
                    sr_output = torch.clamp(sr_output, 0, 1)
                    sr_image = transforms.ToPILImage()(sr_output)
                    
                    # Ensure output size is the target size
                    if sr_image.size != (self.target_size, self.target_size):
                        sr_image = sr_image.resize((self.target_size, self.target_size), Image.BICUBIC)
                    
                    image = sr_image
            
            # Apply final transform for the model
            if self.transform:
                image = self.transform(image)
                
            return image, label, os.path.basename(os.path.dirname(img_path))  # Return image, label, and identity name
    
    # Create datasets
    train_dataset = SimpleDataset(
        train_images, train_labels,
        transform=data_transforms,
        sr_transform=sr_input_transform,
        apply_sr=apply_sr, 
        sr_model=sr_model,
        tf_model=tf_model
    )
    
    val_dataset = SimpleDataset(
        val_images, val_labels,
        transform=data_transforms,
        sr_transform=sr_input_transform,
        apply_sr=apply_sr, 
        sr_model=sr_model,
        tf_model=tf_model
    )
    
    # Create dataloaders with num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    # IMPORTANT: Update num_classes based on the clean identity mapping
    num_classes = len(identity_to_idx)
    print(f"Training with {num_classes} classes")
    
    # Load pre-trained model with updated class count
    model = LightCNN_29Layers_v2(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if it exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        
        # Skip fc2 layer weights as we'll replace them
        if 'fc2.' in name:
            continue
            
        new_state_dict[name] = v
    
    # Load the state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Look at the model structure to understand what layer needs replacement
    print("Model structure before modification:")
    for name, module in model.named_children():
        print(f"  {name}: {module}")
    
    # Replace the final classifier layer (fc2 for LightCNN)
    # The LightCNN_29Layers_v2 model uses fc2 as its final classification layer
    try:
        if hasattr(model, 'fc2'):
            in_features = model.fc2.in_features
            model.fc2 = nn.Linear(in_features, num_classes)
            print(f"Replaced fc2 layer with new shape: [{num_classes}, {in_features}]")
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            print(f"Replaced fc layer with new shape: [{num_classes}, {in_features}]")
        else:
            print("WARNING: Could not find fc or fc2 layer to replace!")
    except AttributeError as e:
        print(f"Error when trying to replace classifier: {e}")
        print("Model structure may be different than expected.")
        # Try to identify the final layer a different way
        found = False
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                print(f"Found linear layer: {name} with shape {module.weight.shape}")
                if name.endswith(('fc2', 'fc')):
                    in_features = module.in_features
                    setattr(model, name, nn.Linear(in_features, num_classes))
                    print(f"Replaced {name} with new shape: [{num_classes}, {in_features}]")
                    found = True
                    break
        if not found:
            print("WARNING: Could not automatically replace the classifier.")
            print("Consider examining the model structure and updating the code.")
    
    # Print model structure for information
    print("Model structure (all layers will be trained):")
    for name, module in model.named_children():
        print(f"  {name}: {module}")

    # No layer freezing - full model finetuning
    print("Training all layers - full model finetuning")

    # Print trainable vs total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Replace optimizer setup
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': 0.0001},  # Lower LR for feature layers
        {'params': model.fc1.parameters(), 'lr': 0.0005},       # Medium LR for fc1 layer
        {'params': model.fc2.parameters(), 'lr': 0.001}         # Higher LR for final layer
    ], momentum=0.9, weight_decay=1e-4)

    # Use step LR scheduler instead of OneCycleLR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Add early stopping logic
    patience = 5
    counter = 0
    best_loss = float('inf')
    
    # Training loop
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for batch_idx, (inputs, labels, _) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    try:
                        # Double-check labels before forward pass
                        if torch.min(labels) < 0 or torch.max(labels) >= num_classes:
                            print(f"WARNING: Label range issue: min={torch.min(labels).item()}, max={torch.max(labels).item()}, num_classes={num_classes}")
                            labels = torch.clamp(labels, 0, num_classes-1)
                            
                        # Get model outputs
                        outputs = model(inputs)
                        
                        # Print output shape for debugging
                        if phase == 'train' and epoch == 0 and outputs is not None:
                            if isinstance(outputs, tuple):
                                print(f"Model returned tuple: {[o.shape for o in outputs]}")
                            else:
                                print(f"Model output shape: {outputs.shape}")
                        
                        # Handle model output format (may return features and logits)
                        if isinstance(outputs, tuple):
                            outputs, features = outputs  # CORRECTED ORDER: logits first, features second
                        
                        # Extra verification
                        if outputs.shape[1] != num_classes:
                            print(f"ERROR: Output classes ({outputs.shape[1]}) don't match num_classes ({num_classes})")
                            
                        # Check for NaNs in outputs
                        if torch.isnan(outputs).any():
                            print("WARNING: NaN values in model outputs")
                            # Replace NaNs with zeros
                            outputs = torch.nan_to_num(outputs)
                        
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Add this inside your training loop after model output
                        if phase == 'train' and epoch == 0 and batch_idx < 2:  # First 2 batches
                            print(f"Batch {batch_idx} sample outputs:")
                            print(f"Output shape: {outputs.shape}")
                            print(f"Output min/max/mean: {outputs.min().item():.3f}/{outputs.max().item():.3f}/{outputs.mean().item():.3f}")
                            print(f"First 3 predictions: {outputs[0, :3]}")
                            print(f"Target labels: {labels[:3]}")
                            
                            # Check for label counts to verify imbalance
                            unique_labels, counts = torch.unique(labels, return_counts=True)
                            print(f"Label distribution in batch: {list(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}")
                        
                        # Debug output at each epoch
                        if batch_idx == 0:  # First batch of each epoch
                            print(f"\nEpoch {epoch+1} - {phase} diagnostics:")
                            print(f"Batch labels: {labels.cpu().numpy()}")
                            print(f"Predictions: {preds.cpu().numpy()}")
                            print(f"Prediction distribution: {torch.bincount(preds, minlength=num_classes)}")
                            
                            # Check for weight stats
                            with torch.no_grad():
                                for name, param in model.named_parameters():
                                    if 'fc2' in name:  # Only check the final layer
                                        print(f"{name} - mean: {param.data.mean():.4f}, std: {param.data.std():.4f}")
                                        print(f"     - grad mean: {param.grad.mean():.4f}, grad std: {param.grad.std():.4f}" 
                                              if param.grad is not None else "     - no grad")
                    except Exception as e:
                        print(f"ERROR in forward pass: {e}")
                        print(f"Input shape: {inputs.shape}, Label range: {labels.min().item()}-{labels.max().item()}")
                        raise e
                        
                # After calculating loss in the training loop

                # Add gradient clipping to prevent exploding gradients
                if phase == 'train':
                    # Backward pass
                    loss.backward()
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate metrics first
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Apply scheduler AFTER calculating loss, and only for validation phase
            if phase == 'val':
                # Use validation loss for ReduceLROnPlateau
                scheduler.step(epoch_loss)
                
                # Early stopping logic
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        break
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save the model
    torch.save({
        'epoch': num_epochs,
        'state_dict': model.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, output_path)
    
    # Return trained model and SR model
    if sr_model is not None:
        return model, sr_model
    else:
        return model, tf_model

if __name__ == "__main__":
    model_path = "/mnt/data/PROJECTS/face-rec-lightcnn/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"
    data_dir = "/mnt/data/PROJECTS/CSRI/Datasets/tinyface/Training_Set"
    output_path = "finetuned_lightcnn_lowres.pth"
    sr_model_path = "pytorch_model_4x.pt"  # Default to TensorFlow model if available
    
    # Get number of classes (identities) in the dataset
    # Modified to access class folders directly (not in a 'train' subdirectory)
    num_classes = len([d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Found {num_classes} identity classes")
    
    # Check if SR model exists
    train_sr = not os.path.exists(sr_model_path)
    
    # Also modify the function call to not expect train/val subdirectories
    # Use a validation split from the same directory
    finetuned_model, sr_model = finetune_lightcnn(
        model_path, 
        data_dir,  # This is now the direct path to class folders
        output_path, 
        num_classes, 
        sr_model_path=sr_model_path,
        train_sr=train_sr,
        num_epochs=15,
        val_split=0.2  # Added validation split parameter
    )
    
    print(f"Model finetuned and saved to {output_path}")