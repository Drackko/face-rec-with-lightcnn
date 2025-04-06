import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from LightCNN.light_cnn import LightCNN_29Layers_v2
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from scipy.spatial.distance import cosine

# Paths and configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = "/mnt/data/PROJECTS/face-rec-lightcnn/yolo/weights/yolo11n-face.pt"
MODEL_PATH = os.path.join(BASE_DIR, "finetuned_lightcnn_lowres.pth")
GALLERY_PATH = os.path.join(BASE_DIR, "gallery.pth")

# Configuration
SIMILARITY_THRESHOLD = 0.6  # Higher = more strict matching
CONFIDENCE_THRESHOLD = 0.5  # For YOLO detections

# Load YOLO model
print(f"Loading YOLO model from {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the gallery
print(f"Loading gallery from {GALLERY_PATH}")
gallery = torch.load(GALLERY_PATH, weights_only=False)
print(f"Loaded gallery with {len(gallery)} identities: {list(gallery.keys())}")

# Load the face recognition model
print(f"Loading LightCNN model from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Determine number of classes for model initialization
num_classes = 2570  # Default from your finetuned model
if 'state_dict' in checkpoint and 'fc2.weight' in checkpoint['state_dict']:
    num_classes = checkpoint['state_dict']['fc2.weight'].shape[0]

model = LightCNN_29Layers_v2(num_classes=num_classes).to(device)

# Load model weights
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("Successfully loaded model weights from state_dict")
else:
    model.load_state_dict(checkpoint, strict=False)
    print("Successfully loaded model weights directly")

model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize as in training
])

def get_face_features(face_img):
    """Extract feature embedding from a face image"""
    face_tensor = transform(face_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(face_tensor)
        
        # Handle model output format (logits, features)
        if isinstance(output, tuple):
            # LightCNN returns (logits, features)
            _, features = output
            return features.squeeze().cpu().numpy()
        else:
            # If model doesn't return features directly
            return output.squeeze().cpu().numpy()

def match_face_to_gallery(face_features, gallery, threshold=0.6):
    """Find the closest match in the gallery"""
    best_match = None
    best_similarity = -1.0
    
    for identity, gallery_features in gallery.items():
        # Convert gallery features to numpy if needed
        if isinstance(gallery_features, torch.Tensor):
            gallery_features = gallery_features.cpu().numpy()
            
        # Calculate cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(face_features, gallery_features)
        
        if similarity > threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = identity
    
    return best_match, best_similarity

# Open video capture
video_path = "/mnt/data/PROJECTS/face-rec-lightcnn/test.mp4"
print(f"Opening video from {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    # Try webcam as fallback
    print("Trying webcam instead...")
    cap = cv2.VideoCapture(0)

# Try webcam as another fallback
if not cap.isOpened():
    print("Trying to open webcam...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam either. Exiting.")
    exit()

print("Starting face recognition loop...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break

    # Detect faces with YOLO
    results = yolo_model(frame)
    
    # Keep track of all detected faces and their recognition results
    detected_faces = []
    
    # First pass: Detect and recognize all faces
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detection_confidence = float(box.conf[0])
            
            if detection_confidence < CONFIDENCE_THRESHOLD:  # Skip low-confidence detections
                continue

            # Extract the detected face
            face = frame[y1:y2, x1:x2]
            if face.size == 0:  # Skip if face region is empty
                continue
                
            try:
                # Convert to PIL image for processing
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                
                # Get face features
                face_features = get_face_features(face_pil)
                
                # Match against gallery
                matched_identity, similarity = match_face_to_gallery(
                    face_features, gallery, threshold=SIMILARITY_THRESHOLD)
                
                # Add face to collection with its coordinates and match result
                detected_faces.append({
                    'coords': (x1, y1, x2, y2),
                    'identity': matched_identity,
                    'similarity': similarity
                })
                    
            except Exception as e:
                print(f"Error processing face: {e}")
                # Still store the face but with no identity
                detected_faces.append({
                    'coords': (x1, y1, x2, y2),
                    'identity': None,
                    'similarity': 0.0
                })
    
    # Draw the results
    for face_data in detected_faces:
        x1, y1, x2, y2 = face_data['coords']
        identity = face_data['identity']
        similarity = face_data['similarity']
        
        if identity:
            # Known person - green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{identity} ({similarity*100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Unknown person - orange box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, "Unknown", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Add frame information
    cv2.putText(frame, f"Faces: {len(detected_faces)} | Threshold: {SIMILARITY_THRESHOLD*100:.0f}%", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show video output with recognition results
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit, 'up'/'down' to adjust threshold
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("User requested exit")
        break
    elif key == 82:  # Up arrow key
        SIMILARITY_THRESHOLD = min(0.95, SIMILARITY_THRESHOLD + 0.05)
        print(f"Increased threshold to {SIMILARITY_THRESHOLD:.2f}")
    elif key == 84:  # Down arrow key
        SIMILARITY_THRESHOLD = max(0.05, SIMILARITY_THRESHOLD - 0.05)
        print(f"Decreased threshold to {SIMILARITY_THRESHOLD:.2f}")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Face recognition completed")