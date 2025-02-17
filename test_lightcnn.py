import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO
from LightCNN.light_cnn import LightCNN_29Layers_v2
import torchvision.transforms as transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR,"yolo", "weights", "yolo11n-face.pt")
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "lightcnn_face_recognizer.pth")

yolo_model = YOLO(YOLO_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["sidd", "sai"]  # Update with your dataset class names
num_classes = len(class_names)


model = LightCNN_29Layers_v2(num_classes=num_classes).to(device)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = yolo_model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # Extract the detected face
            face = frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))  # Convert to grayscale
            face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Add batch dimension

            # Run the face through LightCNN for recognition
            with torch.no_grad():
                logits = model(face_tensor)[1]  # Extract logits

            # Get the predicted class and confidence
            predicted_index = torch.argmax(logits, dim=1).item()
            # Check if the predicted index is within range
            if 0 <= predicted_index < len(class_names):
                predicted_class = class_names[predicted_index]
            else:
                predicted_class = "Unknown"
            recognition_confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_index].item()

            #  Draw Bounding Box & Display Prediction
            label = f"{predicted_class} ({recognition_confidence*100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show video output
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
