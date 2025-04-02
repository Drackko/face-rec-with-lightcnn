# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from pathlib import Path
# from ultralytics import YOLO
# import torchvision.transforms as transforms
# import argparse
# import shutil

# def preprocess_dataset(input_dir, output_dir, min_face_size=80):
#     """
#     Preprocess a dataset of images:
#     1. Detect faces using YOLO
#     2. Crop the faces
#     3. Convert to grayscale
#     4. Resize to 128x128
#     5. Save in class-based directory structure
#     """
#     # Setup model
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo", "weights", "yolo11n-face.pt")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Initialize face detector
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     yolo_model = YOLO(YOLO_MODEL_PATH)
    
#     # Get all the class/person directories
#     person_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
#     print(f"Found {len(person_dirs)} person directories")
    
#     # Process each person's directory
#     for person in person_dirs:
#         person_input_dir = os.path.join(input_dir, person)
#         person_output_dir = os.path.join(output_dir, person)
#         os.makedirs(person_output_dir, exist_ok=True)
        
#         # Get all images for this person
#         image_files = []
#         valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
#         for file in Path(person_input_dir).glob('**/*'):
#             if file.suffix.lower() in valid_extensions:
#                 image_files.append(str(file))
        
#         print(f"Processing {len(image_files)} images for {person}")
        
#         # Process each image
#         face_count = 0
#         for img_path in image_files:
#             # Read the image
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"Failed to read image: {img_path}")
#                 continue
            
#             # Detect faces
#             results = yolo_model(img)
            
#             # Process each detected face
#             for i, result in enumerate(results):
#                 for box in result.boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
#                     # Calculate face size
#                     face_width = x2 - x1
#                     face_height = y2 - y1
                    
#                     # Skip faces that are too small
#                     if face_width < min_face_size or face_height < min_face_size:
#                         continue
                    
#                     # Extract the detected face
#                     face = img[y1:y2, x1:x2]
                    
#                     # Make sure the face region is not empty
#                     if face.size == 0:
#                         continue
                    
#                     # Convert to grayscale
#                     face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    
#                     # Convert to PIL Image 
#                     face_pil = Image.fromarray(face_gray)
                    
#                     # Resize to 128x128
#                     face_pil = face_pil.resize((128, 128), Image.LANCZOS)
                    
#                     # Save the processed face
#                     base_name = os.path.splitext(os.path.basename(img_path))[0]
#                     output_path = os.path.join(person_output_dir, f"{base_name}_face{i}.jpg")
#                     face_pil.save(output_path)
#                     face_count += 1
        
#         print(f"Saved {face_count} processed faces for {person}")
    
#     print(f"Dataset preprocessing complete. Output saved to {output_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Preprocess face dataset for LightCNN training")
#     parser.add_argument("--input", type=str, required=True, help="Input directory containing class folders with images")
#     parser.add_argument("--output", type=str, required=True, help="Output directory for processed dataset")
#     parser.add_argument("--min-face-size", type=int, default=80, help="Minimum face size to include (default: 80 pixels)")
    
#     args = parser.parse_args()
#     preprocess_dataset(args.input, args.output, args.min_face_size)

import os
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path

def preprocess_images(input_dir, output_dir, size=128):
    """
    Preprocess images:
    1. Convert to grayscale
    2. Resize to specified size
    3. Maintain directory structure
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_files = []
    
    print(f"Scanning {input_dir} for images...")
    for ext in valid_extensions:
        all_files.extend(list(Path(input_dir).glob(f'**/*{ext}')))
        all_files.extend(list(Path(input_dir).glob(f'**/*{ext.upper()}')))
    
    print(f"Found {len(all_files)} images to process")
    
    # Process each image
    for img_path in tqdm(all_files, desc="Processing images"):
        # Create relative path for output
        rel_path = img_path.relative_to(input_dir)
        output_path = Path(output_dir) / rel_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            # Read the image using PIL
            img = Image.open(img_path)
            
            # Convert to grayscale
            img_gray = img.convert('L')
            
            # Resize to specified size
            img_resized = img_gray.resize((size, size), Image.LANCZOS)
            
            # Save processed image
            img_resized.save(output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Preprocessing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple preprocessing: grayscale and resize")
    parser.add_argument("--input", type=str, required=True, help="Input directory with images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed images")
    parser.add_argument("--size", type=int, default=128, help="Size to resize images (default: 128)")
    
    args = parser.parse_args()
    preprocess_images(args.input, args.output, args.size)