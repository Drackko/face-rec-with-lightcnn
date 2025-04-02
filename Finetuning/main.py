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

# Import your LightCNN model - adjust import path as needed
from LightCNN.light_cnn import LightCNN_29Layers_v2

# Custom dataset for TinyFaces
class TinyFacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
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
        image = Image.open(img_path).convert('L')  # Convert to grayscale since LightCNN uses grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, os.path.basename(os.path.dirname(img_path))  # Return image, label, and identity name

# Function to finetune the model
def finetune_lightcnn(model_path, data_dir, output_path, num_classes, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    # For low-res images, minimal transforms are better
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),  # LightCNN expects 128x128
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # LightCNN normalization
    ])
    
    # Create datasets
    train_dataset = TinyFacesDataset(os.path.join(data_dir, 'train'), data_transforms)
    val_dataset = TinyFacesDataset(os.path.join(data_dir, 'val'), data_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    # Load pre-trained model
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
        
        # Skip fc layer weights as we'll replace them
        if 'fc.' in name:
            continue
            
        new_state_dict[name] = v
    
    # Load the state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    # Replace the classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Freeze early layers
    for name, param in model.features.named_parameters():
        layer_num = int(name.split('.')[0]) if name.split('.')[0].isdigit() else -1
        if layer_num < 20:  # Freeze first 20 layers
            param.requires_grad = False
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
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
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
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
    
    return model

# Function to create a gallery of face embeddings
def create_face_gallery(model, gallery_data_path, gallery_output_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    gallery_dataset = TinyFacesDataset(gallery_data_path, transform)
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
                gallery[identity] = features[i]
    
    # Save gallery
    torch.save(gallery, gallery_output_path)
    print(f"Gallery saved to {gallery_output_path}")
    return gallery

# Example usage
if __name__ == "__main__":
    model_path = "/mnt/data/PROJECTS/face-rec-lightcnn/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"  # Path to your pretrained model
    data_dir = "/mnt/data/PROJECTS/CSRI/Datasets/tinyface/Training_Set"  # Path to TinyFaces dataset
    output_path = "finetuned_lightcnn_lowres.pth"  # Where to save the finetuned model
    gallery_data_path = "path/to/identifiable_personnel"  # Path to your personnel dataset
    gallery_output_path = "face_gallery.pth"  # Where to save the gallery
    
    # Get number of classes (identities) in your dataset
    num_classes = len([d for d in os.listdir(os.path.join(data_dir, 'train')) if os.path.isdir(os.path.join(data_dir, 'train', d))])
    
    # Finetune the model
    finetuned_model = finetune_lightcnn(model_path, data_dir, output_path, num_classes, num_epochs=15)
    
    # Create gallery
    create_face_gallery(finetuned_model, gallery_data_path, gallery_output_path)