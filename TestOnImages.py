import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from LightCNN.light_cnn import LightCNN_29Layers_v2
import torchvision.transforms as transforms
from PIL import Image
import argparse
from pathlib import Path

def process_images(input_dir, output_dir, confidence_threshold=0.9999):
    # Setup paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo", "weights", "yolo11n-face.pt")
    MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "Oldmodel.pth")
    
    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    class_names = ["sidd","sai"]  # Update with your dataset class names
    num_classes = len(class_names)
    
    model = LightCNN_29Layers_v2(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    image_files = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for file in Path(input_dir).glob('**/*'):
        if file.suffix.lower() in valid_extensions:
            image_files.append(str(file))
    
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        # Get base filename for output
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, base_name)
        
        # Detect faces
        results = yolo_model(img)
        
        # Process each detected face
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection_confidence = float(box.conf[0])
                
                # Extract the detected face
                face = img[y1:y2, x1:x2]
                
                # Handle potential empty face regions
                if face.size == 0:
                    continue
                
                # Prepare face for recognition
                try:
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
                    face_tensor = transform(face_pil).unsqueeze(0).to(device)
                
                    # Run face recognition
                    with torch.no_grad():
                        feature, logits = model(face_tensor)
                    
                    # Get the predicted class and confidence
                    predicted_index = torch.argmax(logits, dim=1).item()
                    recognition_confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_index].item()
                    
                    # Only use class name if effectively 100% match (using 0.9999 for floating point precision)
                    if recognition_confidence >= confidence_threshold and 0 <= predicted_index < len(class_names):
                        predicted_class = class_names[predicted_index]
                    else:
                        predicted_class = "Unknown"
                    
                    # Draw bounding box and label
                    label = f"{predicted_class} ({recognition_confidence*100:.1f}%)"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add a black background for the text for better visibility
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + text_size[0], y1), (0, 0, 0), -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error processing face in {img_path}: {e}")
        
        # Save the output image
        cv2.imwrite(output_path, img)
        print(f"Processed: {img_path} -> {output_path}")
    
    print(f"Finished processing {len(image_files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for face recognition")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed images")
    parser.add_argument("--threshold", type=float, default=0.9999, help="Confidence threshold for recognition (default: 0.9999)")
    
    args = parser.parse_args()
    process_images(args.input, args.output, args.threshold)