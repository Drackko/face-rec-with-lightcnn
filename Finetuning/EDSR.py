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
class MeanShift(nn.Module):
    def __init__(self, rgb_range=1.0, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__()
        
        self.rgb_range = rgb_range
        
        # Handle both RGB and grayscale inputs safely
        if not isinstance(rgb_mean, tuple):
            rgb_mean = (rgb_mean,)
        if not isinstance(rgb_std, tuple):
            rgb_std = (rgb_std,)
        
        std = torch.Tensor(rgb_std)
        self.weight = nn.Parameter(torch.eye(len(rgb_mean)).view(len(rgb_mean), len(rgb_mean), 1, 1) / std.view(len(rgb_mean), 1, 1, 1), requires_grad=False)
        self.bias = nn.Parameter(sign * rgb_range * torch.Tensor(rgb_mean) / std, requires_grad=False)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, self.bias)

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
    def __init__(self, scale=4, n_resblocks=16, n_feats=64, res_scale=1, n_colors=3):
        super(EDSR, self).__init__()
        
        # Define standardization values
        rgb_mean = (0.4488, 0.4371, 0.4040)  # Use (0.5,) for grayscale
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 1.0
        
        # Create MeanShift layers with all required parameters
        self.sub_mean = MeanShift(rgb_range, rgb_mean[:n_colors], rgb_std[:n_colors])
        self.add_mean = MeanShift(rgb_range, rgb_mean[:n_colors], rgb_std[:n_colors], sign=1)
        
        self.head = nn.Sequential(
            nn.Conv2d(n_colors, n_feats, 3, padding=1),
            nn.ReLU(True)
        )

        # define body module
        m_body = [
            ResBlock(n_feats, 3, res_scale=res_scale) 
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))

        # define tail module
        m_tail = [
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, n_colors, 3, padding=1)
        ]

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
        self.sess = None  # Session needs to be initialized
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
            print(f"Failed to load as saved model: {e}")
            # If the above fails, try loading as a frozen graph
            try:
                print(f"Attempting to load as frozen graph from {model_path}")
                with tf.io.gfile.GFile(model_path, "rb") as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                
                self.graph = tf.Graph()
                with self.graph.as_default():
                    tf.import_graph_def(graph_def, name="")
                
                self.sess = tf.compat.v1.Session(graph=self.graph)
                
                # Find input and output nodes
                operations = self.graph.get_operations()
                input_nodes = [op.name for op in operations if 'input' in op.name.lower()]
                output_nodes = [op.name for op in operations if 'output' in op.name.lower()]
                
                print(f"Available operations: {[op.name for op in operations[:10]]}...")
                
                if input_nodes and output_nodes:
                    self.input_node = input_nodes[0] + ':0'
                    self.output_node = output_nodes[0] + ':0'
                else:
                    print("Using placeholder names. Will try common tensor names during processing")
                    self.input_node = "input_tensor:0"
                    self.output_node = "EDSR/output_conv/BiasAdd:0"
                
                print(f"Input node: {self.input_node}")
                print(f"Output node: {self.output_node}")
                print("TensorFlow frozen graph loaded successfully")
            except Exception as e2:
                print(f"Error loading TensorFlow model: {e2}")
                self.model = None
                self.sess = None
    
    def process(self, input_tensor):
        if self.model is None and self.sess is None:
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
            elif self.sess is not None:
                # Try different tensor names if needed
                try:
                    # Get all operations in the graph
                    ops = self.graph.get_operations()
                    input_tensors = [op.name + ':0' for op in ops if 'input' in op.name.lower() or 'placeholder' in op.name.lower()]
                    output_tensors = [op.name + ':0' for op in ops if 'output' in op.name.lower() or 'add_mean' in op.name.lower()]
                    
                    if input_tensors and output_tensors:
                        print(f"Using tensors: Input={input_tensors[0]}, Output={output_tensors[0]}")
                        self.input_node = input_tensors[0]
                        self.output_node = output_tensors[0]
                    
                    # Use frozen graph
                    output_np = self.sess.run(
                        self.output_node,
                        {self.input_node: input_np}
                    )
                except Exception as inner_e:
                    print(f"Failed with specific tensor names: {inner_e}")
                    print("Trying with common tensor names...")
                    # Last resort - try common output tensor names
                    for potential_output in ['output:0', 'NHWC_output:0', 'add_mean/add:0']:
                        try:
                            output_np = self.sess.run(
                                potential_output,
                                {self.input_node: input_np}
                            )
                            print(f"Success with output tensor: {potential_output}")
                            self.output_node = potential_output
                            break
                        except:
                            continue
                    else:
                        raise ValueError("Could not find valid output tensor")
            else:
                raise ValueError("Neither saved model nor session is available")
                
            # Convert back to PyTorch tensor
            return torch.from_numpy(output_np).to(input_tensor.device)
        except Exception as e:
            print(f"Error processing with TensorFlow model: {e}")
            print("Using fallback bicubic upsampling")
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
                   use_fallback=False, target_size=128):
    """
    Process images in a directory using EDSR super-resolution
    
    Args:
        input_dir (str): Directory containing images to process
        output_dir (str): Directory to save processed images (if None, save in same dir)
        model_path (str): Path to pretrained model (.pth for PyTorch, .pb for TensorFlow)
        scale (int): Super-resolution scale factor
        min_size (int): Only process images smaller than this size (if None, process all)
        recursive (bool): Process images in subdirectories
        suffix (str): Suffix to add to processed image filenames
        device (str): Device to use for processing ('cuda' or 'cpu')
        use_fallback (bool): Use fallback bicubic upscaling instead of model
        target_size (int): Target size for the output images (default: 128 for face recognition)
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
                print(f"Loading PyTorch EDSR model from {model_path}")
                # Use grayscale configuration (1 channel)
                sr_model = EDSR(scale=4, n_colors=1).to(device)  # Use 1 channel for grayscale
                
                print(f"EDSR model structure: {sr_model}")
                
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}...")
                
                # Create a new state dict with proper key mapping
                new_state_dict = {}
                for k, v in checkpoint.items():
                    # Handle both module prefix and channel count mismatch
                    name = k[7:] if k.startswith('module.') else k
                    
                    # For MeanShift layers which might have channel dimension mismatch
                    if ('sub_mean' in name or 'add_mean' in name) and 'weight' in name:
                        # Original is likely [3,3,1,1] but we need [1,1,1,1]
                        if v.size(0) == 3 and sr_model.state_dict()[name].size(0) == 1:
                            # Take only the first channel
                            print(f"Adapting {name} from shape {v.shape} to {sr_model.state_dict()[name].shape}")
                            v = v[0:1, 0:1, :, :]
                    
                    if name in sr_model.state_dict():
                        if v.shape != sr_model.state_dict()[name].shape:
                            print(f"Shape mismatch for {name}: checkpoint {v.shape} vs model {sr_model.state_dict()[name].shape}")
                            continue
                        new_state_dict[name] = v
                
                # Load adapted weights
                sr_model.load_state_dict(new_state_dict, strict=False)
                print("EDSR model loaded successfully!")
            except Exception as e:
                print(f"Failed to load EDSR model: {e}")
                sr_model = None
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
    
    # Calculate input size to achieve target_size after super-resolution
    base_size = target_size // scale
    print(f"Using input size of {base_size}x{base_size} to achieve {target_size}x{target_size} after {scale}x upscaling")
    
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
            
            # Resize to base_size to ensure consistent output size
            # First, preserve aspect ratio while making the smallest dimension equal to base_size
            width, height = img.size
            if width < height:
                new_width = base_size
                new_height = int(height * (base_size / width))
            else:
                new_height = base_size
                new_width = int(width * (base_size / height))
                
            # Resize with aspect ratio preserved
            img_resized = img.resize((new_width, new_height), Image.BICUBIC)
            
            # Then center crop to base_size x base_size
            left = (new_width - base_size) // 2
            top = (new_height - base_size) // 2
            right = left + base_size
            bottom = top + base_size
            
            img_cropped = img_resized.crop((left, top, right, bottom))
            
            # Prepare image for model
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input_tensor = transform(img_cropped).unsqueeze(0).to(device)
            
            # Process with model or fallback
            if 'sr_model' in locals() and sr_model is not None:
                with torch.no_grad():
                    # Convert grayscale to 3 channels if needed
                    if input_tensor.size(1) == 1 and sr_model.sub_mean.weight.size(0) == 3:
                        input_tensor = input_tensor.repeat(1, 3, 1, 1)
                    output_tensor = sr_model(input_tensor)
                    # If model output is RGB but we need grayscale, take first channel
                    if output_tensor.size(1) == 3:
                        output_tensor = output_tensor[:, 0:1, :, :]
            else:
                # Use fallback
                print("Model not loaded properly, using fallback")
                output_tensor = torch.nn.functional.interpolate(
                    input_tensor, scale_factor=scale, mode='bicubic', align_corners=False
                )
            
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
            
            # Verify size is target_size x target_size
            if output_image.size != (target_size, target_size):
                print(f"Warning: Output size {output_image.size} doesn't match target size ({target_size}x{target_size}). Resizing.")
                output_image = output_image.resize((target_size, target_size), Image.BICUBIC)
            
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
                # Use original image and upscaled image for comparison
                original_resized = img.resize((target_size, target_size), Image.BICUBIC)
                # Create side-by-side image
                comparison = Image.new('L', (target_size * 2, target_size))
                comparison.paste(original_resized, (0, 0))
                comparison.paste(output_image, (target_size, 0))
                comparison.save(comparison_path)
                print(f"Saved comparison to {comparison_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    print("Processing complete!")

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, apply_sr=False, sr_model=None, device=None, min_size=128):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.apply_sr = apply_sr
        self.sr_model = sr_model
        self.device = device
        self.min_size = min_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Determine if super-resolution is needed based on image size
        needs_sr = False
        if self.apply_sr and self.sr_model is not None:
            width, height = image.size
            if width < self.min_size or height < self.min_size:
                needs_sr = True
        
        if needs_sr:
            # For SR model, convert grayscale to proper format
            # Create a single-channel tensor
            sr_input = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Single channel normalization
            ])(image)
            
            # Apply super-resolution
            with torch.no_grad():
                sr_input = sr_input.unsqueeze(0).to(self.device)
                sr_output = self.sr_model(sr_input)
                # Denormalize output
                sr_output = sr_output.squeeze(0).cpu()
                sr_output = (sr_output + 1) / 2
                sr_output = torch.clamp(sr_output, 0, 1)
                sr_image = transforms.ToPILImage()(sr_output)
                
                # Ensure output size is 128x128
                if sr_image.size != (128, 128):
                    sr_image = sr_image.resize((128, 128), Image.LANCZOS)
                image = sr_image
        
        # Apply final transform for the model
        if self.transform:
            image = self.transform(image)
        
        return image, label, os.path.basename(os.path.dirname(img_path))

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
    parser.add_argument("--target-size", type=int, default=128, 
                        help="Target size for output images (default: 128 for face recognition)")
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
        use_fallback=args.fallback,
        target_size=args.target_size
    )
