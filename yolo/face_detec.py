import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import argparse

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
                
                # Add some margin (10% of width/height)
                w, h = x2 - x1, y2 - y1
                margin_x, margin_y = int(0.1 * w), int(0.1 * h)
                
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
                
                # Create unique filename
                timestamp = int(time.time() * 1000)
                filename = f"{video_name}_frame{frame_count}_face{i}_{timestamp}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Save the face
                cv2.imwrite(filepath, face)
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
    parser.add_argument("--output", default="mnt/data/PROJECTS/face-rec-lightcnn/base_dataset/raw_faces", help="Output directory for cropped faces")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    if args.video:
        detect_and_save_faces(args.video, args.output, args.conf)
    elif args.video_dir:
        process_videos(args.video_dir, args.output, args.conf)
    else:
        print("Please provide either --video or --video_dir argument")