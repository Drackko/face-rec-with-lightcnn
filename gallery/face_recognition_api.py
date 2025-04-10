import torch
import cv2
import numpy as np
import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import torchvision.transforms as transforms
from LightCNN.light_cnn import LightCNN_29Layers_v2
from scipy.spatial.distance import cosine
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Face Recognition API", description="API for face recognition using LightCNN")

# Global variables to hold models (loaded once at startup)
face_model = None
yolo_model = None
gallery = None
device = None
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class RecognitionResult(BaseModel):
    face_id: int
    identity: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]

class RecognitionResponse(BaseModel):
    image_base64: str
    faces: List[RecognitionResult]
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    """Load models and gallery at startup"""
    global face_model, yolo_model, gallery, device
    
    # Path configurations (customize these)
    model_path = "/mnt/data/PROJECTS/face-rec-lightcnn/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"
    gallery_path = "/mnt/data/PROJECTS/face-rec-lightcnn/face_gallery.pth"
    yolo_path = "/mnt/data/PROJECTS/face-rec-lightcnn/yolo/weights/yolo11n-face.pt"
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference")
    
    # Load face recognition model
    face_model = LightCNN_29Layers_v2(num_classes=100)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Filter out the fc2 layer parameters
        if 'state_dict' in checkpoint:
            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if 'fc2' in k:
                    continue
                new_k = k.replace("module.", "")
                new_state_dict[new_k] = v
        else:
            new_state_dict = {}
            for k, v in checkpoint.items():
                if 'fc2' in k:
                    continue
                new_k = k.replace("module.", "")
                new_state_dict[new_k] = v
        
        face_model.load_state_dict(new_state_dict, strict=False)
        face_model = face_model.to(device)
        face_model.eval()
        print("✓ Face recognition model loaded")
    except Exception as e:
        print(f"❌ Error loading face model: {e}")
        raise RuntimeError(f"Failed to load face recognition model: {e}")
    
    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_path)
        print("✓ YOLO face detection model loaded")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    # Load gallery
    try:
        gallery = torch.load(gallery_path)
        print(f"✓ Gallery loaded with {len(gallery)} identities")
    except Exception as e:
        print(f"❌ Error loading gallery: {e}")
        raise RuntimeError(f"Failed to load gallery: {e}")

def process_image(image_bytes, threshold=0.6):
    """
    Process an image for face recognition
    
    Args:
        image_bytes: Raw image bytes
        threshold: Recognition confidence threshold
        
    Returns:
        Tuple of (processed_image, recognition_results)
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Process each face
    result_img = img.copy()
    faces = []
    face_id = 0
    
    # Detect faces using YOLO
    results = yolo_model(img)
    
    # Process each detected face
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
            
            # Convert BGR to grayscale PIL image
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            
            # Get face tensor and extract embedding
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                _, embedding = face_model(face_tensor)
                face_embedding = embedding.cpu().squeeze().numpy()
            
            # Find best match
            best_match = "Unknown"
            best_score = -1
            
            for identity, gallery_embedding in gallery.items():
                # Calculate cosine similarity
                similarity = 1 - cosine(face_embedding, gallery_embedding)
                
                if similarity > threshold and similarity > best_score:
                    best_score = similarity
                    best_match = identity
            
            # Draw result on image
            if best_match != "Unknown":
                # Known identity - green box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{best_match} ({best_score:.2f})"
                cv2.putText(result_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Unknown - red box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(result_img, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add to results
            face_id += 1
            confidence = float(best_score) if best_score > 0 else 0.0
            faces.append(RecognitionResult(
                face_id=face_id,
                identity=best_match,
                confidence=confidence,
                bbox=[int(x1), int(y1), int(x2), int(y2)]
            ))
    
    # Encode image to base64 to return in response
    _, buffer = cv2.imencode('.jpg', result_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64, faces

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_faces(
    file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    """
    Recognize faces in an uploaded image
    
    Args:
        file: Image file upload
        threshold: Confidence threshold for recognition (0.0-1.0)
        
    Returns:
        Processed image and recognition results
    """
    # Check if models are loaded
    if face_model is None or yolo_model is None or gallery is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Please check server logs.")
    
    # Validate threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
    
    try:
        # Read file content
        contents = await file.read()
        
        # Process image
        img_base64, faces = process_image(contents, threshold)
        
        # Return response
        return RecognitionResponse(
            image_base64=img_base64,
            faces=faces,
            status="success",
            message=f"Recognized {len(faces)} faces"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_upload_page():
    """Return a simple HTML form for testing the API"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            input, button { margin: 10px 0; }
            #result { margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; }
            table, th, td { border: 1px solid #ddd; }
            th, td { padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            #resultImage { max-width: 100%; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Face Recognition Service</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <h3>Upload an image for face recognition</h3>
            <div>
                <label for="file">Select image:</label>
                <input type="file" id="file" name="file" accept="image/*" required>
            </div>
            <div>
                <label for="threshold">Recognition threshold (0.0-1.0):</label>
                <input type="number" id="threshold" name="threshold" min="0" max="1" step="0.05" value="0.6">
            </div>
            <button type="submit">Recognize Faces</button>
        </form>
        
        <div id="result" style="display: none;">
            <h3>Recognition Results</h3>
            <div id="status"></div>
            <table id="facesTable">
                <thead>
                    <tr>
                        <th>Face ID</th>
                        <th>Identity</th>
                        <th>Confidence</th>
                        <th>Bounding Box</th>
                    </tr>
                </thead>
                <tbody id="facesBody"></tbody>
            </table>
            <h3>Processed Image</h3>
            <img id="resultImage" src="" alt="Processed image">
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('file');
                const threshold = document.getElementById('threshold').value;
                
                formData.append('file', fileInput.files[0]);
                formData.append('threshold', threshold);
                
                try {
                    const response = await fetch('/recognize', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    // Display results
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('status').textContent = result.message;
                    
                    // Clear previous results
                    const facesBody = document.getElementById('facesBody');
                    facesBody.innerHTML = '';
                    
                    // Add new face results
                    result.faces.forEach(face => {
                        const row = document.createElement('tr');
                        
                        const idCell = document.createElement('td');
                        idCell.textContent = face.face_id;
                        
                        const identityCell = document.createElement('td');
                        identityCell.textContent = face.identity;
                        
                        const confidenceCell = document.createElement('td');
                        confidenceCell.textContent = (face.confidence * 100).toFixed(2) + '%';
                        
                        const bboxCell = document.createElement('td');
                        bboxCell.textContent = `[${face.bbox.join(', ')}]`;
                        
                        row.appendChild(idCell);
                        row.appendChild(identityCell);
                        row.appendChild(confidenceCell);
                        row.appendChild(bboxCell);
                        
                        facesBody.appendChild(row);
                    });
                    
                    // Display processed image
                    document.getElementById('resultImage').src = 'data:image/jpeg;base64,' + result.image_base64;
                    
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error processing image. See console for details.');
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("face_recognition_api:app", host="0.0.0.0", port=8000, reload=True)