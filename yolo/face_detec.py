import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import argparse
import dlib  # For facial landmark detection

def normalize_face(face_img, target_size=(128, 128), use_landmarks=True):
    """
    Normalize and enhance a face image to make it more suitable for LightCNN
    and landmark detection.
    
    Args:
        face_img: The face image to process
        target_size: Output size (default 128x128 for LightCNN)
        use_landmarks: Whether to attempt face alignment using landmarks
    
    Returns:
        Normalized face image
    """
    # Convert to grayscale if not already
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Optional: Face alignment using facial landmarks
    normalized_face = denoised
    if use_landmarks:
        try:
            # Initialize dlib's face detector and landmark predictor
            predictor_path = "shape_predictor_68_face_landmarks.dat"  # You need to download this
            if os.path.exists(predictor_path):
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor(predictor_path)
                
                # Detect face and landmarks
                dlib_rect = detector(denoised, 1)[0]
                landmarks = predictor(denoised, dlib_rect)
                
                # Get coordinates for eyes
                left_eye = np.array([(landmarks.part(36).x + landmarks.part(39).x) // 2,
                                    (landmarks.part(36).y + landmarks.part(39).y) // 2])
                right_eye = np.array([(landmarks.part(42).x + landmarks.part(45).x) // 2,
                                    (landmarks.part(42).y + landmarks.part(45).y) // 2])
                
                # Calculate angle and perform alignment
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Rotate to align eyes horizontally
                center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                normalized_face = cv2.warpAffine(denoised, M, (denoised.shape[1], denoised.shape[0]))
        except Exception as e:
            print(f"Warning: Face alignment failed - {e}")
    
    # Resize to target size
    normalized_face = cv2.resize(normalized_face, target_size)
    
    # Final gaussian blur to slightly smooth the image (remove if it affects landmarks)
    # normalized_face = cv2.GaussianBlur(normalized_face, (3, 3), 0)
    
    return normalized_face

def detect_and_save_faces(video_path, output_dir="/mnt/data/PROJECTS/face-rec-lightcnn/base_dataset/raw_faces", confidence=0.7):
    """
    Detect faces in a video using YOLOv8 and save them to the output directory.
    Process at original resolution to maximize quality.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save cropped faces
        confidence (float): Confidence threshold for face detection
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model for face detection
    model = YOLO("weights/yolo11n-face.pt")  # Use appropriate YOLO face detection model
    
    # Get video filename for naming
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get original video dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video at original resolution: {original_width}x{original_height}")
    
    frame_count = 0
    faces_saved = 0
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame to save compute
            continue
        
        # Detect faces on the original resolution 
        # Pass the image size parameter to avoid auto-resizing
        results = model(frame, conf=confidence, imgsz=(original_height, original_width))
        
        # Process detections
        if results[0].boxes.data.shape[0] > 0:
            for i, box in enumerate(results[0].boxes.data):
                x1, y1, x2, y2, conf, _ = box.cpu().numpy()
                
                if conf < confidence:
                    continue
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                w, h = x2 - x1, y2 - y1
                margin_x, margin_y = int(0.2 * w), int(0.2 * h)
                
                # Ensure coordinates are within frame boundaries
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(frame.shape[1], x2 + margin_x)
                y2 = min(frame.shape[0], y2 + margin_y)
                
                # Crop face
                face = frame[y1:y2, x1:x2]
                
                # Skip if face crop is empty
                if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                    continue
                
                # Apply face normalization
                normalized_face = normalize_face(face, target_size=(128, 128))
                
                # Create unique filename
                timestamp = int(time.time() * 1000)
                filename = f"{video_name}_frame{frame_count}_face{i}_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Save the normalized face
                cv2.imwrite(filepath, normalized_face)
                faces_saved += 1
                
                if faces_saved % 10 == 0:  # Print status every 10 faces
                    print(f"Saved {faces_saved} faces so far...")
    
    # Release video
    cap.release()
    print(f"Processed {frame_count} frames from {video_path}, saved {faces_saved} faces")

def process_videos(video_dir, output_dir="/mnt/data/PROJECTS/face-rec-lightcnn/base_dataset/raw_faces", confidence=0.7):
    """
    Process all video files in a directory
    
    Args:
        video_dir (str): Directory containing video files
        output_dir (str): Directory to save cropped faces
        confidence (float): Confidence threshold for detection
    """
    for file in os.listdir(video_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, file)
            print(f"Processing {video_path}...")
            detect_and_save_faces(video_path, output_dir, confidence)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop faces from videos using YOLO at original resolution")
    parser.add_argument("--video", help="Path to a single video file")
    parser.add_argument("--video_dir", help="Directory containing video files")
    parser.add_argument("--output", default="/mnt/data/PROJECTS/face-rec-lightcnn/base_dataset/raw_faces", help="Output directory for cropped faces")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    if args.video:
        detect_and_save_faces(args.video, args.output, args.conf)
    elif args.video_dir:
        process_videos(args.video_dir, args.output, args.conf)
    else:
        print("Please provide either --video or --video_dir argument")