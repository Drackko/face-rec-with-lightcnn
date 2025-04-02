import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from LightCNN.light_cnn import LightCNN_29Layers_v2
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR,"yolo", "weights", "yolo11n-face.pt")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "test.pth")

# Configuration
CONFIDENCE_THRESHOLD = 0.95

# Load YOLO model
print(f"Loading YOLO model from {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved model checkpoint
print(f"Loading LightCNN model from {MODEL_WEIGHTS_PATH}")
checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)

# Get class mapping from checkpoint
if 'idx_to_class' in checkpoint:
    idx_to_class = checkpoint['idx_to_class']
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Loaded {len(class_names)} classes from checkpoint")
    # Print first few classes for verification
    for i in range(min(5, len(class_names))):
        print(f"  Class {i}: {class_names[i]}")
    if len(class_names) > 5:
        print(f"  ... and {len(class_names)-5} more classes")
else:
    # Fallback to hardcoded class names
    class_names = ["mithun","sai"]
    print(f"Using hardcoded class names: {class_names}")

# Initialize model
num_classes = len(class_names)
model = LightCNN_29Layers_v2(num_classes=num_classes).to(device)

# Load model weights
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
    print("Successfully loaded model weights from state_dict")
else:
    # Try loading directly (older format)
    try:
        model.load_state_dict(checkpoint)
        print("Successfully loaded model weights directly")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Using model with random initialization")

model.eval()

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Open video capture
video_path = "/mnt/data/PROJECTS/face-rec-lightcnn/test2.mp4"
print(f"Opening video from {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    # Try webcam as fallback
    print("Trying webcam instead...")
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
            
            if detection_confidence < 0.4:  # Skip low-confidence detections
                continue

            # Extract the detected face
            face = frame[y1:y2, x1:x2]
            if face.size == 0:  # Skip if face region is empty
                continue
                
            try:
                # Convert to grayscale and prepare for model
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                # Run the face through LightCNN for recognition
                with torch.no_grad():
                    output = model(face_tensor)
                    
                    # Handle model output format (features, logits)
                    if isinstance(output, tuple):
                        features, logits = output
                    else:
                        logits = output
                        
                    # Get probabilities with softmax
                    probs = torch.nn.functional.softmax(logits, dim=1)[0]
                    
                    # Store top 3 predictions for this face
                    top_values, top_indices = torch.topk(probs, min(3, len(probs)))
                    
                    predictions = []
                    for i in range(len(top_indices)):
                        idx = top_indices[i].item()
                        conf = top_values[i].item()
                        if idx < len(class_names):
                            predictions.append((class_names[idx], conf, idx))
                
                # Add face to collection with its coordinates and predictions
                detected_faces.append({
                    'coords': (x1, y1, x2, y2),
                    'predictions': predictions
                })
                    
            except Exception as e:
                print(f"Error processing face: {e}")
                # Still store the face but with no predictions
                detected_faces.append({
                    'coords': (x1, y1, x2, y2),
                    'predictions': []
                })
    
    # Second pass: Find highest confidence for each class
    # For each class label, only the face with highest confidence will get that label
    class_to_best_face = {}  # Maps class name to (face_index, confidence)
    
    for face_idx, face_data in enumerate(detected_faces):
        for class_name, confidence, class_idx in face_data['predictions']:
            # Only consider predictions above threshold
            if confidence < CONFIDENCE_THRESHOLD:
                continue
                
            # Is this the best confidence we've seen for this class?
            if (class_name not in class_to_best_face or 
                confidence > class_to_best_face[class_name][1]):
                class_to_best_face[class_name] = (face_idx, confidence)
    
    # Third pass: Draw the results
    for face_idx, face_data in enumerate(detected_faces):
        x1, y1, x2, y2 = face_data['coords']
        assigned_label = None
        assigned_confidence = 0
        
        # Check if this face is the best match for any class
        for class_name, (best_face_idx, confidence) in class_to_best_face.items():
            if face_idx == best_face_idx:
                assigned_label = class_name
                assigned_confidence = confidence
                break
        
        # Draw bounding box and label
        if assigned_label:
            # High confidence - green
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{assigned_label} ({assigned_confidence*100:.1f}%)"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # No confident assignment - orange
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, "Unknown", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Add frame information
    cv2.putText(frame, f"Faces: {len(detected_faces)} | Identified: {len(class_to_best_face)}", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show video output with recognition results
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("User requested exit")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Face recognition completed")