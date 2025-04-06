import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import argparse
import shutil
import warnings

# Import LightCNN model
from LightCNN.light_cnn import LightCNN_29Layers_v2

# Import EDSR components
from EDSR import EDSR, TFModelHandler

# Custom dataset for face images with super-resolution support
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, apply_sr=False, sr_model=None, 
                 tf_model=None, min_size=60, target_size=128):
        self.root_dir = root_dir
        self.transform = transform
        self.apply_sr = apply_sr
        self.sr_model = sr_model  # PyTorch EDSR model
        self.tf_model = tf_model  # TensorFlow EDSR model
        self.min_size = min_size  # Min size threshold to apply SR
        self.target_size = target_size  # Final target size for the model
        self.images = []
        self.labels = []
        self.identities = []
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
                        self.identities.append(identity)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        identity = self.identities[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Determine if super-resolution is needed based on image size
        needs_sr = False
        if self.apply_sr and (self.sr_model is not None or self.tf_model is not None):
            width, height = image.size
            if width < self.min_size or height < self.min_size:
                needs_sr = True
                
        if needs_sr:
            # Prepare for SR
            sr_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            sr_input = sr_transform(image).unsqueeze(0).to(self.device)
            
            # Apply super-resolution
            with torch.no_grad():
                if self.sr_model is not None:
                    sr_output = self.sr_model(sr_input)
                elif self.tf_model is not None:
                    sr_output = self.tf_model.process(sr_input)
                
                # Convert back to PIL
                # Denormalize: from [-1,1] to [0,1] range
                sr_output = sr_output.squeeze(0).cpu()
                sr_output = (sr_output + 1) / 2
                sr_output = torch.clamp(sr_output, 0, 1)
                sr_image = transforms.ToPILImage()(sr_output)
                
                # Ensure output size matches target size
                if sr_image.size != (self.target_size, self.target_size):
                    sr_image = sr_image.resize((self.target_size, self.target_size), Image.BICUBIC)
                
                image = sr_image
        
        # Apply final transform for the model
        if self.transform:
            image = self.transform(image)
            
        return image, label, identity

# Load super-resolution model based on file extension
def load_sr_model(model_path, scale=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sr_model = None
    tf_model = None
    
    if model_path and os.path.exists(model_path):
        # Determine model type based on extension
        if model_path.endswith('.pb') or os.path.isdir(model_path):
            # TensorFlow model
            try:
                print(f"Loading TensorFlow EDSR model from {model_path}")
                tf_model = TFModelHandler(model_path, scale=scale)
            except Exception as e:
                print(f"Failed to load TensorFlow model: {e}")
        else:
            # PyTorch model
            try:
                print(f"Loading PyTorch EDSR model from {model_path}")
                sr_model = EDSR(scale=scale).to(device)
                
                try:
                    sr_model.load_state_dict(torch.load(model_path, map_location=device))
                except Exception as e:
                    print(f"Error loading with weights_only=True: {e}")
                    print("Attempting to load with weights_only=False")
                    warnings.warn("Loading model with weights_only=False can execute arbitrary code")
                    
                    try:
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            sr_model.load_state_dict(checkpoint['state_dict'])
                        else:
                            sr_model.load_state_dict(checkpoint)
                    except Exception as e2:
                        print(f"Error loading model: {e2}")
                        sr_model = None
            except Exception as e:
                print(f"Failed to load PyTorch model: {e}")
                sr_model = None
    
    # Set model to eval mode if available
    if sr_model is not None:
        sr_model.eval()
        
    return sr_model, tf_model

# Function to create a gallery of face embeddings
def create_face_gallery(model_path, gallery_data_path, gallery_output_path, sr_model_path=None, num_classes=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint first to extract model information
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get num_classes from checkpoint if possible
    if num_classes is None:
        # Try to determine number of classes from the checkpoint
        if 'state_dict' in checkpoint:
            for key, value in checkpoint['state_dict'].items():
                if 'fc2.weight' in key:
                    num_classes = value.shape[0]
                    print(f"Detected {num_classes} classes from checkpoint")
                    break
        else:
            for key, value in checkpoint.items():
                if 'fc2.weight' in key:
                    num_classes = value.shape[0]
                    print(f"Detected {num_classes} classes from checkpoint")
                    break
        
        # If still not determined, count directories
        if num_classes is None:
            num_classes = len([d for d in os.listdir(gallery_data_path) 
                            if os.path.isdir(os.path.join(gallery_data_path, d))])
            print(f"Using {num_classes} classes from directory count")
    
    # Load face recognition model with correct number of classes
    print(f"Initializing LightCNN model with {num_classes} classes")
    model = LightCNN_29Layers_v2(num_classes=num_classes)
    
    # Load model weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Load super-resolution model if provided
    sr_model, tf_model = load_sr_model(sr_model_path)
    apply_sr = (sr_model is not None or tf_model is not None)
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    gallery_dataset = FaceDataset(
        gallery_data_path, 
        transform=transform,
        apply_sr=apply_sr,
        sr_model=sr_model,
        tf_model=tf_model
    )
    
    gallery_loader = DataLoader(gallery_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Dictionary to store embeddings
    gallery = {}
    
    with torch.no_grad():
        for images, _, identities in gallery_loader:
            images = images.to(device)
            
            # Extract features (embeddings)
            output = model(images)
            
            # Handle different model output formats
            if isinstance(output, tuple):
                # LightCNN returns (out, fc) where fc is the feature embedding
                _, features = output  # IMPORTANT: Second element is the feature vector
                features = features.data.cpu().numpy()
            else:
                # If the model doesn't have a dedicated feature extraction method
                # Use the output directly
                print("Warning: Model doesn't return features directly, using output as features")
                features = output.data.cpu().numpy()
            
            # Store in gallery
            for i, identity in enumerate(identities):
                if identity in gallery:
                    # Average with existing embedding if already present
                    gallery[identity] = (gallery[identity] + features[i]) / 2
                else:
                    gallery[identity] = features[i]
    
    # Save gallery
    torch.save(gallery, gallery_output_path)
    print(f"Gallery created with {len(gallery)} identities and saved to {gallery_output_path}")
    return gallery

# Function to update an existing gallery with new personnel
def update_gallery(model_path, existing_gallery_path, new_personnel_path, output_gallery_path, sr_model_path=None, num_classes=None):
    # Load existing gallery
    if os.path.exists(existing_gallery_path):
        existing_gallery = torch.load(existing_gallery_path)
        print(f"Loaded existing gallery with {len(existing_gallery)} identities")
    else:
        existing_gallery = {}
        print("No existing gallery found, creating new one")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if num_classes is None:
        # Use the number of classes from the original gallery, or count from the data
        all_identities = set(existing_gallery.keys())
        for identity in os.listdir(new_personnel_path):
            if os.path.isdir(os.path.join(new_personnel_path, identity)):
                all_identities.add(identity)
        num_classes = len(all_identities)
    
    model = LightCNN_29Layers_v2(num_classes=num_classes)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load super-resolution model if provided
    sr_model, tf_model = load_sr_model(sr_model_path)
    apply_sr = (sr_model is not None or tf_model is not None)
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Process new personnel
    new_personnel_dataset = FaceDataset(
        new_personnel_path,
        transform=transform,
        apply_sr=apply_sr,
        sr_model=sr_model,
        tf_model=tf_model
    )
    
    new_personnel_loader = DataLoader(new_personnel_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    new_embeddings = {}
    with torch.no_grad():
        for images, _, identities in new_personnel_loader:
            images = images.to(device)
            
            # Extract features (embeddings)
            output = model(images)
            
            # Handle different model output formats
            if isinstance(output, tuple):
                # LightCNN returns (out, fc) where fc is the feature embedding
                _, features = output  # IMPORTANT: Second element is the feature vector
                features = features.data.cpu().numpy()
            else:
                features = output.data.cpu().numpy()
            
            # Store in new embeddings
            for i, identity in enumerate(identities):
                if identity in new_embeddings:
                    # Average with existing embedding if already present
                    new_embeddings[identity] = (new_embeddings[identity] + features[i]) / 2
                else:
                    new_embeddings[identity] = features[i]
    
    # Update gallery
    updated_gallery = existing_gallery.copy()
    for identity, embedding in new_embeddings.items():
        updated_gallery[identity] = embedding
    
    # Save updated gallery
    torch.save(updated_gallery, output_gallery_path)
    print(f"Updated gallery with {len(new_embeddings)} new identities")
    print(f"Total identities in gallery: {len(updated_gallery)}")
    print(f"Gallery saved to {output_gallery_path}")
    
    return updated_gallery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Gallery Manager with Super-Resolution')
    parser.add_argument('--mode', type=str, required=True, choices=['create', 'update'],
                        help='Mode: create a new gallery or update an existing one')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the finetuned LightCNN model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the face data directory')
    parser.add_argument('--gallery_path', type=str, required=True, 
                        help='Path to save or load the gallery')
    parser.add_argument('--output_path', type=str,
                        help='Path to save the updated gallery (for update mode)')
    parser.add_argument('--sr_model_path', type=str, default=None,
                        help='Path to the super-resolution model (.pth or .pb)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of identity classes (optional)')
    
    args = parser.parse_args()
    
    if args.mode == 'create':
        create_face_gallery(
            args.model_path, 
            args.data_path, 
            args.gallery_path, 
            sr_model_path=args.sr_model_path,
            num_classes=args.num_classes
        )
    elif args.mode == 'update':
        if not args.output_path:
            args.output_path = args.gallery_path
        update_gallery(
            args.model_path, 
            args.gallery_path, 
            args.data_path, 
            args.output_path, 
            sr_model_path=args.sr_model_path,
            num_classes=args.num_classes
        )