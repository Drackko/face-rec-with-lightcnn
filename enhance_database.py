import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

# Define the EDSR model architecture
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, res_scale=1):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        m = []
        if scale == 4:  # x4 upsampling
            m.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1))
            m.append(nn.PixelShuffle(2))
        elif scale == 2:  # x2 upsampling
            m.append(nn.Conv2d(n_feats, n_feats * 4, 3, padding=1))
            m.append(nn.PixelShuffle(2))
        else:  # x3 upsampling
            m.append(nn.Conv2d(n_feats, n_feats * 9, 3, padding=1))
            m.append(nn.PixelShuffle(3))
            
        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4):
        super(EDSR, self).__init__()
        
        kernel_size = 3
        rgb_range = 255
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        
        # Define head
        m_head = [
            nn.Conv2d(3, n_feats, kernel_size, padding=kernel_size//2)
        ]
        
        # Define body
        m_body = [
            ResBlock(n_feats, kernel_size) for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
        
        # Define tail
        m_tail = [
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, 3, kernel_size, padding=kernel_size//2)
        ]
        
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, sign=1)
        
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

def enhance_image(img_path, model, target_size=None):
    """Enhance a single image using local EDSR model"""
    # Load image using PIL
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Could not open image: {str(e)}")
    
    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
    
    # Process with model
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert back to PIL image
    output = output.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    output = output.astype(np.uint8)
    enhanced_img = Image.fromarray(output)
    
    # Resize to target size if specified
    if target_size:
        enhanced_img = enhanced_img.resize((target_size, target_size), Image.LANCZOS)
    
    return enhanced_img

def load_model(model_path, scale=4):
    """Load PyTorch EDSR model"""
    print(f"Loading model from {model_path}")
    
    # Create model
    model = EDSR(scale=scale)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Handle different state dict formats
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    # Remove the 'module.' prefix from state dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
        new_state_dict[name] = v
    
    # Load state dict
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model

def enhance_database(input_db_dir, output_db_dir, model_path, scale=4, target_size=128, min_size=None):
    """Enhance all face images in a database directory using EDSR"""
    # Create output directory
    os.makedirs(output_db_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path, scale)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Process each person folder
    person_folders = [d for d in os.listdir(input_db_dir) 
                     if os.path.isdir(os.path.join(input_db_dir, d))]
    
    print(f"Found {len(person_folders)} person folders in the database")
    
    for i, person in enumerate(person_folders):
        print(f"Processing person {i+1}/{len(person_folders)}: {person}")
        
        # Create folders
        person_input_dir = os.path.join(input_db_dir, person)
        person_output_dir = os.path.join(output_db_dir, person)
        os.makedirs(person_output_dir, exist_ok=True)
        
        # Get all images
        image_files = [f for f in os.listdir(person_input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"  Found {len(image_files)} images for {person}")
        processed_count = 0
        
        for img_file in image_files:
            input_path = os.path.join(person_input_dir, img_file)
            output_path = os.path.join(person_output_dir, img_file)
            
            # Check if image needs processing
            if min_size:
                try:
                    with Image.open(input_path) as img:
                        width, height = img.size
                    if min(width, height) >= min_size:
                        # Just copy the image
                        with open(input_path, 'rb') as src:
                            with open(output_path, 'wb') as dst:
                                dst.write(src.read())
                        processed_count += 1
                        continue
                except Exception:
                    pass
            
            # Process image
            try:
                enhanced_img = enhance_image(input_path, model, target_size)
                enhanced_img.save(output_path)
                processed_count += 1
                
                # Print progress
                if processed_count % 5 == 0:
                    print(f"  Processed {processed_count}/{len(image_files)} images...")
                
            except Exception as e:
                print(f"  Error processing {input_path}: {str(e)}")
                # Copy original as fallback
                try:
                    with open(input_path, 'rb') as src:
                        with open(output_path, 'wb') as dst:
                            dst.write(src.read())
                except Exception:
                    print(f"  Could not copy original image as fallback")
                processed_count += 1
    
    print(f"Database enhancement complete. Enhanced database saved to {output_db_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance face database using EDSR super-resolution")
    parser.add_argument("--input", required=True, help="Input database directory containing person folders")
    parser.add_argument("--output", required=True, help="Output directory for enhanced database")
    parser.add_argument("--model", required=True, help="Path to EDSR PyTorch model file (.pt)")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4],
                        help="Upscaling factor (2, 3, or 4)")
    parser.add_argument("--target-size", type=int, default=128, help="Target image size (default: 128)")
    parser.add_argument("--min-size", type=int, default=None, 
                        help="Only enhance images smaller than this size (default: enhance all)")
    
    args = parser.parse_args()
    
    enhance_database(
        args.input, 
        args.output, 
        args.model,
        args.scale,
        args.target_size,
        args.min_size
    )