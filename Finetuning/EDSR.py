import torch
import torch.nn as nn
import os
import argparse
from PIL import Image
import math
from torchvision import transforms
import glob
import numpy as np

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

def process_images(input_dir, output_dir=None, model_path=None, scale=4, 
                   min_size=None, recursive=False, suffix="_sr", device=None):
    """
    Process images in a directory using EDSR super-resolution
    
    Args:
        input_dir (str): Directory containing images to process
        output_dir (str): Directory to save processed images (if None, save in same dir)
        model_path (str): Path to pretrained EDSR model
        scale (int): Super-resolution scale factor
        min_size (int): Only process images smaller than this size (if None, process all)
        recursive (bool): Process images in subdirectories
        suffix (str): Suffix to add to processed image filenames
        device (str): Device to use for processing ('cuda' or 'cpu')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Initialize model
    model = EDSR(scale=scale).to(device)
    
    # Load pretrained model if available
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Using untrained model (results may be poor)")
    
    model.eval()
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all image files
    if recursive:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    else:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for img_path in image_files:
        try:
            # Load and convert to grayscale
            img = Image.open(img_path).convert('L')
            orig_size = img.size
            
            # Skip large images if min_size is specified
            if min_size and (orig_size[0] >= min_size and orig_size[1] >= min_size):
                print(f"Skipping {img_path} (size: {orig_size})")
                continue
            
            # Prepare image for model
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Process with EDSR
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            # Convert back to PIL image
            output_tensor = output_tensor.clamp(0, 1)
            output_tensor = output_tensor * 2 - 1  # Denormalize
            output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
            
            # Determine output path
            if output_dir:
                rel_path = os.path.relpath(img_path, input_dir)
                output_subdir = os.path.dirname(os.path.join(output_dir, rel_path))
                if output_subdir and not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                filename, ext = os.path.splitext(os.path.basename(img_path))
                output_path = os.path.join(output_subdir, f"{filename}{suffix}{ext}")
            else:
                filename, ext = os.path.splitext(img_path)
                output_path = f"{filename}{suffix}{ext}"
            
            # Save the processed image
            output_image.save(output_path)
            print(f"Processed: {img_path} ({orig_size}) -> {output_path} ({output_image.size})")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDSR Super-Resolution for images")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--model", type=str, default="EDSR_x4.pb", help="Path to pretrained EDSR model")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor (default: 4)")
    parser.add_argument("--min-size", type=int, default=None, 
                        help="Only process images smaller than this size (default: process all)")
    parser.add_argument("--recursive", action="store_true", help="Process images in subdirectories")
    parser.add_argument("--suffix", type=str, default="_sr", help="Suffix for processed image filenames")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use for processing (default: cuda if available, else cpu)")
    
    args = parser.parse_args()
    
    process_images(
        input_dir=args.input,
        output_dir=args.output, 
        model_path=args.model,
        scale=args.scale,
        min_size=args.min_size,
        recursive=args.recursive,
        suffix=args.suffix,
        device=args.device
    )