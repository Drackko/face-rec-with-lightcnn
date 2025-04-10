import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import argparse
from LightCNN.light_cnn import LightCNN_29Layers_v2
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# Consistent image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_model(model_path):
    """Load LightCNN model with correct architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for computation")
    
    # Initialize model with arbitrary number of classes (we only need embeddings)
    model = LightCNN_29Layers_v2(num_classes=100)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Filter out the fc2 layer parameters to avoid dimension mismatch
    if 'state_dict' in checkpoint:
        # Remove "module." prefix and fc2 layer parameters
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            # Skip fc2 layer parameters
            if 'fc2' in k:
                continue
            # Remove module prefix if present
            new_k = k.replace("module.", "")
            new_state_dict[new_k] = v
    else:
        # Direct state dict without 'state_dict' key
        new_state_dict = {}
        for k, v in checkpoint.items():
            if 'fc2' in k:
                continue
            new_k = k.replace("module.", "")
            new_state_dict[new_k] = v
    
    # Load the filtered state dict
    model.load_state_dict(new_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    return model, device

def extract_embedding(model, img_path, device):
    """Extract a face embedding from an image using LightCNN"""
    try:
        # Load and transform image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            # LightCNN returns a tuple (output, features)
            _, embedding = model(img_tensor)
            return embedding.cpu().squeeze().numpy()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def create_gallery(model_path, data_dir, output_path):
    """Create a face recognition gallery from preprocessed face images"""
    # Load model
    model, device = load_model(model_path)
    
    # Create gallery dictionary
    gallery = {}
    
    # Process each identity folder
    identities = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(identities)} identities")
    
    for identity in tqdm(identities, desc="Processing identities"):
        identity_dir = os.path.join(data_dir, identity)
        
        # Get all images for this identity
        image_files = [f for f in os.listdir(identity_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"Warning: No images found for {identity}")
            continue
        
        # Extract embeddings for all images
        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(identity_dir, img_file)
            embedding = extract_embedding(model, img_path, device)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            print(f"Warning: No valid embeddings extracted for {identity}")
            continue
        
        # Average embeddings to get a single representation
        avg_embedding = np.mean(embeddings, axis=0)
        gallery[identity] = avg_embedding
    
    print(f"Gallery created with {len(gallery)} identities")
    
    # Save gallery
    torch.save(gallery, output_path)
    print(f"Gallery saved to {output_path}")
    return gallery

def update_gallery(model_path, gallery_path, new_data_dir, output_path=None):
    """Update an existing gallery with new identities"""
    if output_path is None:
        output_path = gallery_path
        
    # Load existing gallery
    if os.path.exists(gallery_path):
        existing_gallery = torch.load(gallery_path)
        print(f"Loaded existing gallery with {len(existing_gallery)} identities")
    else:
        existing_gallery = {}
        print("No existing gallery found, creating new one")
    
    # Load model
    model, device = load_model(model_path)
    
    # Process new identities
    identities = [d for d in os.listdir(new_data_dir) if os.path.isdir(os.path.join(new_data_dir, d))]
    print(f"Found {len(identities)} new identities to process")
    
    # Create updated gallery
    updated_gallery = existing_gallery.copy()
    
    for identity in tqdm(identities, desc="Processing new identities"):
        identity_dir = os.path.join(new_data_dir, identity)
        
        # Skip if identity already in gallery (unless forced to update)
        if identity in updated_gallery:
            print(f"Identity {identity} already in gallery, updating...")
        
        # Get all images for this identity
        image_files = [f for f in os.listdir(identity_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"Warning: No images found for {identity}")
            continue
        
        # Extract embeddings for all images
        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(identity_dir, img_file)
            embedding = extract_embedding(model, img_path, device)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            print(f"Warning: No valid embeddings extracted for {identity}")
            continue
        
        # Average embeddings to get a single representation
        avg_embedding = np.mean(embeddings, axis=0)
        updated_gallery[identity] = avg_embedding
    
    # Save updated gallery
    torch.save(updated_gallery, output_path)
    print(f"Updated gallery saved to {output_path}")
    print(f"Gallery now contains {len(updated_gallery)} identities")
    return updated_gallery

def test_gallery(model_path, gallery_path, image_path, threshold=0.6, yolo_path=None):
    """Test gallery recognition on a single image"""
    # Load model and gallery
    model, device = load_model(model_path)
    gallery = torch.load(gallery_path)
    print(f"Loaded gallery with {len(gallery)} identities")
    
    # Load YOLO for face detection if provided
    if yolo_path:
        yolo_model = YOLO(yolo_path)
    else:
        yolo_model = None
        
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
        
    faces = []
    
    # Extract faces using YOLO if available
    if yolo_model:
        results = yolo_model(img)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add padding around face
                h, w = img.shape[:2]
                face_w = x2 - x1
                face_h = y2 - y1
                pad_x = int(face_w * 0.2)
                pad_y = int(face_h * 0.2)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                face = img[y1:y2, x1:x2]
                faces.append((face, (x1, y1, x2, y2)))
    else:
        # If no YOLO, use whole image as face
        faces.append((img, (0, 0, img.shape[1], img.shape[0])))
    
    # Process each face
    result_img = img.copy()
    for face, coords in faces:
        # Convert BGR to grayscale PIL image
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
        
        # Get face tensor
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            _, embedding = model(face_tensor)
            face_embedding = embedding.cpu().squeeze().numpy()
        
        # Find best match
        best_match = None
        best_score = -1
        
        for identity, gallery_embedding in gallery.items():
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(face_embedding, gallery_embedding)
            
            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = identity
        
        # Draw result
        x1, y1, x2, y2 = coords
        if best_match:
            # Known identity - green box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{best_match} ({best_score:.2f})"
            cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Unknown - red box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(result_img, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Show result
    # result_img = cv2.resize(result_img, (1280, 720))
    # cv2.imshow("Gallery Recognition", result_img)
    cv2.imwrite("result.jpg", result_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face gallery manager")
    parser.add_argument("--mode", choices=["create", "update", "test"], required=True,
                        help="Operation mode: create, update, or test gallery")
    parser.add_argument("--model", default="/mnt/data/PROJECTS/face-rec-lightcnn/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar", 
                        help="Path to LightCNN model")
    parser.add_argument("--gallery", required=True, help="Path to face gallery")
    parser.add_argument("--data", help="Path to face data directory (for create/update)")
    parser.add_argument("--output", help="Output path (for update)")
    parser.add_argument("--image", help="Path to test image (for test)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    parser.add_argument("--yolo", default="/mnt/data/PROJECTS/face-rec-lightcnn/yolo/weights/yolo11n-face.pt", 
                        help="Path to YOLO face detection model")
    
    args = parser.parse_args()
    
    if args.mode == "create":
        if not args.data:
            print("Error: --data required for create mode")
        else:
            create_gallery(args.model, args.data, args.gallery)
    
    elif args.mode == "update":
        if not args.data:
            print("Error: --data required for update mode")
        else:
            update_gallery(args.model, args.gallery, args.data, args.output)
    
    elif args.mode == "test":
        if not args.image:
            print("Error: --image required for test mode")
        else:
            test_gallery(args.model, args.gallery, args.image, args.threshold, args.yolo)