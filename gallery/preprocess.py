import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def detect_and_crop_faces(img_path, yolo_model, conf_threshold=0.5, padding=0.0):
    """
    Detect faces in an image and crop them out
    
    Args:
        img_path: Path to image file
        yolo_model: YOLO face detection model
        conf_threshold: Confidence threshold for detections
        padding: Optional padding around face as a percentage of face size
        
    Returns:
        List of cropped face images as numpy arrays
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return []
        
    # Run face detection
    results = yolo_model(img, conf=conf_threshold)
    
    faces = []
    
    if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        img_height, img_width = img.shape[:2]
        
        # Process each detected face
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate padding
            if padding > 0:
                face_width = x2 - x1
                face_height = y2 - y1
                pad_x = int(face_width * padding)
                pad_y = int(face_height * padding)
                
                # Add padding with bounds checking
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(img_width, x2 + pad_x)
                y2 = min(img_height, y2 + pad_y)
            
            # Crop face region
            face = img[y1:y2, x1:x2]
            if face.size > 0:  # Ensure we have a valid crop
                faces.append(face)
    
    return faces

def preprocess_for_lcnn(face_img, target_size=(128, 128)):
    """
    Process a face image for LightCNN:
    - Convert to grayscale
    - Resize to target size
    - Normalize
    
    Args:
        face_img: Face image as numpy array (BGR)
        target_size: Output image size (default: 128x128)
        
    Returns:
        Processed image as numpy array
    """
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return resized

def process_dataset(input_dir, output_dir, yolo_model_path, conf_threshold=0.5, padding=0.2):
    """
    Process an entire face dataset to create a LightCNN optimized version
    
    Args:
        input_dir: Input dataset directory (with identity subfolders)
        output_dir: Output directory for processed dataset
        yolo_model_path: Path to YOLO face detection model
        conf_threshold: Confidence threshold for face detection
        padding: Padding around detected faces (percentage of face size)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO face detection model from {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    
    # Get all identity folders
    identities = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print(f"Found {len(identities)} identity folders")
    
    # Process each identity folder
    for identity in tqdm(identities, desc="Processing identity folders"):
        # Create output identity folder
        identity_input_dir = os.path.join(input_dir, identity)
        identity_output_dir = os.path.join(output_dir, identity)
        os.makedirs(identity_output_dir, exist_ok=True)
        
        # Get all images for this identity
        image_files = [f for f in os.listdir(identity_input_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"Warning: No images found for {identity}")
            continue
        
        # Process each image
        face_count = 0
        for img_file in image_files:
            img_path = os.path.join(identity_input_dir, img_file)
            
            # Detect and crop faces
            faces = detect_and_crop_faces(img_path, yolo_model, conf_threshold, padding)
            
            if not faces:
                # No faces detected in this image
                continue
            
            # Process and save each detected face
            for i, face in enumerate(faces):
                # Process for LightCNN
                processed_face = preprocess_for_lcnn(face)
                
                # Generate output filename
                base_name = os.path.splitext(img_file)[0]
                if len(faces) > 1:
                    # If multiple faces in one image, add face index
                    output_name = f"{base_name}_face{i+1}.jpg"
                else:
                    output_name = f"{base_name}.jpg"
                
                output_path = os.path.join(identity_output_dir, output_name)
                
                # Save processed face
                cv2.imwrite(output_path, processed_face)
                face_count += 1
        
        print(f"Processed {face_count} faces for identity: {identity}")
    
    print(f"Dataset processing complete. Optimized dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a face dataset for LightCNN")
    parser.add_argument("--input", required=True, help="Input dataset directory (with identity subfolders)")
    parser.add_argument("--output", required=True, help="Output directory for processed dataset")
    parser.add_argument("--yolo", default="/mnt/data/PROJECTS/face-rec-lightcnn/yolo/weights/yolo11n-face.pt", 
                        help="Path to YOLO face detection model")
    parser.add_argument("--conf", type=float, default=0.5, 
                        help="Confidence threshold for face detection")
    parser.add_argument("--padding", type=float, default=0.2, 
                        help="Padding around detected faces (percentage of face size)")
    
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.yolo, args.conf, args.padding)