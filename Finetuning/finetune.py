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

# Import LightCNN model
from LightCNN.light_cnn import LightCNN_29Layers_v2

# EDSR Model Definition
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(1, 1, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(1).view(1, 1, 1, 1) / std.view(1, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act:
                    m.append(nn.ReLU(True))
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act:
                m.append(nn.ReLU(True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, scale=4, n_resblocks=16, n_feats=64, res_scale=1):
        super(EDSR, self).__init__()
        
        # RGB mean for input normalization (grayscale for LightCNN)
        rgb_mean = (0.5,)
        rgb_std = (1.0,)
        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)
        
        # define head module
        m_head = [nn.Conv2d(1, n_feats, 3, padding=1)]

        # define body module
        m_body = [
            ResBlock(n_feats, 3, res_scale=res_scale) 
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))

        # define tail module
        m_tail = [
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, 1, 3, padding=1)
        ]

        self.add_mean = MeanShift(1, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

# Custom dataset for TinyFaces with EDSR super-resolution
class TinyFacesDataset(Dataset):
    def __init__(self, root_dir, transform=None, sr_transform=None, apply_sr=False, 
                 sr_model=None, min_size=60, target_size=128):
        self.root_dir = root_dir
        self.transform = transform
        self.sr_transform = sr_transform  # Transform for SR input
        self.apply_sr = apply_sr
        self.sr_model = sr_model
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
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Determine if super-resolution is needed based on image size
        needs_sr = False
        if self.apply_sr and self.sr_model is not None:
            width, height = image.size
            if width < self.min_size or height < self.min_size:
                needs_sr = True
                
        if needs_sr:
            # Apply initial transform for SR input
            if self.sr_transform:
                sr_input = self.sr_transform(image)
            else:
                # Default SR input transform
                sr_input = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
                ])(image)
                
            # Apply super-resolution
            with torch.no_grad():
                sr_input = sr_input.unsqueeze(0).to(self.device)
                sr_output = self.sr_model(sr_input)
                # Convert back to PIL for further processing
                sr_image = transforms.ToPILImage()(sr_output.squeeze(0).cpu())
                image = sr_image
        
        # Apply final transform for the model
        if self.transform:
            image = self.transform(image)
            
        return image, label, os.path.basename(os.path.dirname(img_path))  # Return image, label, and identity name

# Function to train EDSR model for super-resolution
def train_edsr(data_dir, output_path, scale=4, num_epochs=50, batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for EDSR training")
    
    # Initialize EDSR model
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
    
    # Collect image paths
    train_images = []
    train_dir = os.path.join(data_dir, 'train')
    for identity in os.listdir(train_dir):
        identity_dir = os.path.join(train_dir, identity)
        if os.path.isdir(identity_dir):
            for img_name in os.listdir(identity_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    train_images.append(os.path.join(identity_dir, img_name))
    
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
                      sr_model_path=None, train_sr=False, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Handle EDSR model
    sr_model = None
    if sr_model_path and os.path.exists(sr_model_path):
        print(f"Loading EDSR model from {sr_model_path}")
        sr_model = EDSR(scale=4).to(device)
        sr_model.load_state_dict(torch.load(sr_model_path, map_location=device))
        sr_model.eval()
    elif train_sr:
        print("Training EDSR model for super-resolution...")
        sr_output_path = sr_model_path if sr_model_path else "edsr_trained.pth"
        sr_model = train_edsr(data_dir, sr_output_path)
        sr_model.eval()
    else:
        print("EDSR model not provided or trained. Proceeding without super-resolution.")
    
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
    
    # Create datasets with SR if available
    apply_sr = sr_model is not None
    train_dataset = TinyFacesDataset(
        os.path.join(data_dir, 'train'), 
        transform=data_transforms,
        sr_transform=sr_input_transform,
        apply_sr=apply_sr, 
        sr_model=sr_model
    )
    
    val_dataset = TinyFacesDataset(
        os.path.join(data_dir, 'val'), 
        transform=data_transforms,
        sr_transform=sr_input_transform,
        apply_sr=apply_sr, 
        sr_model=sr_model
    )
    
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
    
    return model, sr_model

if __name__ == "__main__":
    model_path = "/mnt/data/PROJECTS/face-rec-lightcnn/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"
    data_dir = "/mnt/data/PROJECTS/CSRI/Datasets/tinyface/Training_Set"
    output_path = "finetuned_lightcnn_lowres.pth"
    sr_model_path = "edsr_model.pth"
    
    # Get number of classes (identities) in the dataset
    num_classes = len([d for d in os.listdir(os.path.join(data_dir, 'train')) 
                      if os.path.isdir(os.path.join(data_dir, 'train', d))])
    
    # Check if EDSR model exists
    train_sr = not os.path.exists(sr_model_path)
    
    # Finetune the model with EDSR super-resolution
    finetuned_model, sr_model = finetune_lightcnn(
        model_path, 
        data_dir, 
        output_path, 
        num_classes, 
        sr_model_path=sr_model_path,
        train_sr=train_sr,
        num_epochs=15
    )
    
    print(f"Model finetuned and saved to {output_path}")
    if train_sr:
        print(f"EDSR model trained and saved to {sr_model_path}")