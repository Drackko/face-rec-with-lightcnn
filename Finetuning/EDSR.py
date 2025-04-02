import torch
import torch.nn as nn
import os
import argparse
from PIL import Image
import math
from torchvision import transforms
import glob
import numpy as np
import warnings
from io import BytesIO

# Try to import TensorFlow
TF_AVAILABLE = False
try:
    import tensorflow as tf
    print("TensorFlow is available, version:", tf.__version__)
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow is not available. Only PyTorch models can be used.")

# EDSR Model Definition (PyTorch)
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

# TensorFlow model handler
class TFModelHandler:
    def __init__(self, model_path, scale=4):
        self.model = None
        self.scale = scale
        self.load_model(model_path)
        
    def load_model(self, model_path):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot load .pb model.")
            
        try:
            # Load the TensorFlow model
            self.model = tf.saved_model.load(model_path)
            print(f"TensorFlow model loaded from {model_path}")
            
            # Try to get the serving function
            self.infer = self.model.signatures["serving_default"]
            print("Model input:", self.infer.structured_input_signature)
            print("Model output:", self.infer.structured_outputs)
        except Exception as e:
            # If the above fails, try loading as a frozen graph
            try:
                with tf.io.gfile.GFile(model_path, "rb") as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                
                self.graph = tf.Graph()
                with self.graph.as_default():
                    tf.import_graph_def(graph_def, name="")
                
                self.sess = tf.compat.v1.Session(graph=self.graph)
                
                # Find input and output nodes
                input_nodes = [n.name for n in graph_def.node if "input" in n.name.lower()]
                output_nodes = [n.name for n in graph_def.node if "output" in n.name.lower()]
                
                if input_nodes and output_nodes:
                    self.input_node = input_nodes[0]
                    self.output_node = output_nodes[0]
                    print(f"Model input node: {self.input_node}")
                    print(f"Model output node: {self.output_node}")
                else:
                    print("Could not identify input/output nodes. Using default names.")
                    self.input_node = "input:0"
                    self.output_node = "output:0"
                
                print(f"TensorFlow frozen graph loaded from {model_path}")
            except Exception as e2:
                print(f"Error loading TensorFlow model: {e2}")
                self.model = None
    
    def process(self, input_tensor):
        if self.model is None:
            raise ValueError("No TensorFlow model loaded")
        
        # Convert PyTorch tensor to NumPy array
        input_np = input_tensor.cpu().numpy()
        
        try:
            # Try using the serving signature
            if hasattr(self, 'infer'):
                # Adapt based on your model's input requirements
                tf_input = tf.convert_to_tensor(input_np)
                result = self.infer(tf_input)
                # Get the output tensor (adjust key as needed)
                output_key = list(result.keys())[0]
                output_np = result[output_key].numpy()
            else:
                # Use frozen graph
                output_np = self.sess.run(
                    self.output_node,
                    {self.input_node: input_np}
                )
                
            # Convert back to PyTorch tensor
            return torch.from_numpy(output_np).to(input_tensor.device)
        except Exception as e:
            print(f"Error processing with TensorFlow model: {e}")
            # Fallback to simple bicubic upsampling
            return self._fallback_upscale(input_tensor)
    
    def _fallback_upscale(self, input_tensor):
        # Simple fallback using PyTorch interpolation
        print("Using fallback bicubic upscaling")
        return torch.nn.functional.interpolate(
            input_tensor, 
            scale_factor=self.scale, 
            mode='bicubic', 
            align_corners=False
        )

def process_images(input_dir, output_dir=None, model_path=None, scale=4, 
                   min_size=None, recursive=False, suffix="_sr", device=None,
                   use_fallback=False):
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
        use_fallback (bool): Use fallback bicubic upscaling instead of model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Determine model type and load accordingly
    model = None
    tf_model = None
    
    if use_fallback:
        print("Using fallback bicubic upscaling instead of model")
    elif model_path and os.path.exists(model_path):
        if model_path.endswith('.pb') or os.path.isdir(model_path):
            # TensorFlow model
            if TF_AVAILABLE:
                try:
                    tf_model = TFModelHandler(model_path, scale)
                    print(f"Using TensorFlow model from {model_path}")
                except Exception as e:
                    print(f"Error loading TensorFlow model: {e}")
                    print("Using fallback bicubic upscaling")
                    use_fallback = True
            else:
                print("TensorFlow not available. Using fallback bicubic upscaling")
                use_fallback = True
        else:
            # PyTorch model
            try:
                model = EDSR(scale=scale).to(device)
                
                try:
                    # Try loading with weights_only=True (safer)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    print(f"Loaded PyTorch model from {model_path}")
                except Exception as e:
                    print(f"Error loading model with weights_only=True: {e}")
                    print("Attempting to load with weights_only=False (only do this if the model is from a trusted source)")
                    warnings.warn("Loading model with weights_only=False can execute arbitrary code if the file is malicious")
                    
                    try:
                        # Try with weights_only=False (potential security risk)
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        
                        # Handle different checkpoint formats
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        
                        print(f"Successfully loaded model from {model_path}")
                    except Exception as e2:
                        print(f"Error loading model with weights_only=False: {e2}")
                        print("Using fallback bicubic upscaling")
                        use_fallback = True
            except Exception as e:
                print(f"Error initializing PyTorch model: {e}")
                print("Using fallback bicubic upscaling")
                use_fallback = True
    else:
        if not use_fallback:
            print("No model file found. Using fallback bicubic upscaling")
            use_fallback = True
    
    if model is not None:
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
            
            # Process with model or fallback
            if use_fallback:
                # Simple bicubic upscaling
                output_tensor = torch.nn.functional.interpolate(
                    input_tensor, 
                    scale_factor=scale, 
                    mode='bicubic', 
                    align_corners=False
                )
            elif tf_model is not None:
                # Use TensorFlow model
                output_tensor = tf_model.process(input_tensor)
            else:
                # Use PyTorch model
                with torch.no_grad():
                    output_tensor = model(input_tensor)
            
            # Show tensor statistics for debugging
            min_val = output_tensor.min().item()
            max_val = output_tensor.max().item()
            mean_val = output_tensor.mean().item()
            print(f"Output tensor stats - Min: {min_val}, Max: {max_val}, Mean: {mean_val}")
            
            # Convert back to PIL image
            output_tensor = output_tensor.squeeze(0).cpu()
            # Proper denormalization: from [-1,1] to [0,1] range
            output_tensor = (output_tensor + 1) / 2
            # Ensure values are in valid range
            output_tensor = torch.clamp(output_tensor, 0, 1)
            # Convert to PIL image
            output_image = transforms.ToPILImage()(output_tensor)
            
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
            
            # Save a side-by-side comparison for visual inspection
            if output_dir:
                comparison_path = os.path.join(output_subdir, f"{filename}_compare{ext}")
                # Resize original to match SR image size for fair comparison
                resized_original = img.resize((output_image.width, output_image.height), Image.BICUBIC)
                # Create side-by-side image
                comparison = Image.new('L', (output_image.width * 2, output_image.height))
                comparison.paste(resized_original, (0, 0))
                comparison.paste(output_image, (output_image.width, 0))
                comparison.save(comparison_path)
                print(f"Saved comparison to {comparison_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDSR Super-Resolution for images")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--model", type=str, default="EDSR_x4.pb", help="Path to pretrained model (.pth for PyTorch, .pb for TensorFlow)")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor (default: 4)")
    parser.add_argument("--min-size", type=int, default=None, 
                        help="Only process images smaller than this size (default: process all)")
    parser.add_argument("--recursive", action="store_true", help="Process images in subdirectories")
    parser.add_argument("--suffix", type=str, default="_sr", help="Suffix for processed image filenames")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use for processing (default: cuda if available, else cpu)")
    parser.add_argument("--fallback", action="store_true", help="Use bicubic upscaling instead of model")
    
    args = parser.parse_args()
    
    process_images(
        input_dir=args.input,
        output_dir=args.output, 
        model_path=args.model,
        scale=args.scale,
        min_size=args.min_size,
        recursive=args.recursive,
        suffix=args.suffix,
        device=args.device,
        use_fallback=args.fallback
    )