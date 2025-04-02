import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import argparse
import shutil

# Import LightCNN model
from LightCNN.light_cnn import LightCNN_29Layers_v2

# Custom dataset for face images
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.identities = []
        
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
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, identity

# Function to create a gallery of face embeddings
def create_face_gallery(model_path, gallery_data_path, gallery_output_path, num_classes=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if num_classes is None:
        # Count number of identity folders
        num_classes = len([d for d in os.listdir(gallery_data_path) 
                          if os.path.isdir(os.path.join(gallery_data_path, d))])
    
    model = LightCNN_29Layers_v2(num_classes=num_classes)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    gallery_dataset = FaceDataset(gallery_data_path, transform)
    gallery_loader = DataLoader(gallery_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Dictionary to store embeddings
    gallery = {}
    
    with torch.no_grad():
        for images, _, identities in gallery_loader:
            images = images.to(device)
            
            # Extract features (embeddings)
            features = model.extract_feature(images).data.cpu().numpy()
            
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
def update_gallery(model_path, existing_gallery_path, new_personnel_path, output_gallery_path, num_classes=None):
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
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Process new personnel
    new_personnel_dataset = FaceDataset(new_personnel_path, transform)
    new_personnel_loader = DataLoader(new_personnel_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    new_embeddings = {}
    with torch.no_grad():
        for images, _, identities in new_personnel_loader:
            images = images.to(device)
            
            # Extract features (embeddings)
            features = model.extract_feature(images).data.cpu().numpy()
            
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
    parser = argparse.ArgumentParser(description='Face Gallery Manager')
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
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of identity classes (optional)')
    
    args = parser.parse_args()
    
    if args.mode == 'create':
        create_face_gallery(args.model_path, args.data_path, args.gallery_path, args.num_classes)
    elif args.mode == 'update':
        if not args.output_path:
            args.output_path = args.gallery_path
        update_gallery(args.model_path, args.gallery_path, args.data_path, args.output_path, args.num_classes)