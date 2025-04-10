import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import argparse
from PIL import Image
from tqdm import tqdm

def preprocess_face_for_lightcnn(face_img, target_size=(128, 128)):
    """
    Process a face image for LightCNN:
    - Convert to grayscale
    - Resize to target size
    
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

def extract_faces_from_video(video_path, output_base_dir, 
                            face_model_path="yolo/weights/yolo11n-face.pt",
                            sample_rate=10, frames_per_sample=3,
                            face_confidence=0.5, face_padding=0.2):
    """
    Extract faces from video, preprocess them, and save to output directory.
    
    Args:
        video_path: Path to video file
        output_base_dir: Base directory for outputs
        face_model_path: Path to YOLO face detection model
        sample_rate: Process 1 batch of frames per this many frames (10 = 10%)
        frames_per_sample: Number of consecutive frames to process in each batch
        face_confidence: Confidence threshold for face detection
        face_padding: Padding around detected faces (percentage of face size)
    """
    # Get video name without extension for the output folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO face detection model
    print(f"Loading YOLO face detection model from {face_model_path}")
    face_model = YOLO(face_model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, Duration: {duration:.2f}s, Frames: {frame_count}")
    print(f"Sampling: {frames_per_sample} frames every {sample_rate} frames")
    
    # Initialize counters
    processed_frame_count = 0
    faces_detected = 0
    faces_saved = 0
    current_frame = 0
    
    # Create progress bar
    pbar = tqdm(total=frame_count)
    
    # Process video
    while True:
        # Handle frame sampling
        if current_frame % sample_rate == 0:
            # Process next frames_per_sample frames
            for i in range(frames_per_sample):
                # Make sure we don't go beyond the end of the video
                if current_frame + i >= frame_count:
                    break
                
                # Set position and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + i)
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame_count += 1
                
                # Detect faces
                results = face_model(frame, conf=face_confidence)
                
                # Process each detected face
                if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    for j, box in enumerate(results[0].boxes):
                        faces_detected += 1
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Skip if below confidence threshold
                        if conf < face_confidence:
                            continue
                        
                        # Add padding around face
                        face_width = x2 - x1
                        face_height = y2 - y1
                        pad_x = int(face_width * face_padding)
                        pad_y = int(face_height * face_padding)
                        
                        # Ensure coordinates are within frame boundaries
                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(frame.shape[1], x2 + pad_x)
                        y2 = min(frame.shape[0], y2 + pad_y)
                        
                        # Crop face
                        face = frame[y1:y2, x1:x2]
                        
                        # Skip if face crop is empty
                        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                            continue
                        
                        # Preprocess face for LightCNN
                        processed_face = preprocess_face_for_lightcnn(face)
                        
                        # Create unique filename
                        timestamp = int(time.time() * 1000)
                        filename = f"{video_name}_frame{current_frame+i}_face{j}_{timestamp}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        # Save processed face
                        cv2.imwrite(filepath, processed_face)
                        faces_saved += 1
            
            # Update progress bar every sample
            pbar.update(sample_rate)
        
        # Move to next batch
        current_frame += sample_rate
        if current_frame >= frame_count:
            break
    
    # Close resources
    cap.release()
    pbar.close()
    
    # Print summary
    print("\n" + "="*50)
    print(f"FACE EXTRACTION SUMMARY FOR: {video_path}")
    print("="*50)
    print(f"Processed {processed_frame_count} frames ({processed_frame_count/frame_count*100:.1f}% of total)")
    print(f"Total faces detected: {faces_detected}")
    print(f"Total processed faces saved: {faces_saved}")
    print(f"Output directory: {output_dir}")
    print("="*50)
    
    return faces_saved

def process_videos_in_directory(video_dir, output_base_dir, 
                              face_model_path="yolo/weights/yolo11n-face.pt",
                              sample_rate=10, frames_per_sample=3,
                              face_confidence=0.5, face_padding=0.2):
    """
    Process all videos in a directory
    """
    # Get all video files
    video_files = []
    for file in os.listdir(video_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(os.path.join(video_dir, file))
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    total_faces = 0
    for video_path in video_files:
        faces = extract_faces_from_video(
            video_path, 
            output_base_dir,
            face_model_path, 
            sample_rate, 
            frames_per_sample,
            face_confidence, 
            face_padding
        )
        total_faces += faces
    
    print(f"Processed {len(video_files)} videos, extracted {total_faces} faces total")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and preprocess faces from videos")
    parser.add_argument("--video", help="Path to a single video file")
    parser.add_argument("--video_dir", help="Directory containing multiple video files")
    parser.add_argument("--output", default="processed_faces", 
                        help="Base output directory for processed faces")
    parser.add_argument("--yolo", default="yolo/weights/yolo11n-face.pt", 
                        help="Path to YOLO face detection model")
    parser.add_argument("--sample_rate", type=int, default=10, 
                        help="Process 1 batch of frames per this many frames (default: 10)")
    parser.add_argument("--frames_per_batch", type=int, default=3, 
                        help="Number of consecutive frames to process in each batch (default: 3)")
    parser.add_argument("--face_conf", type=float, default=0.5, 
                        help="Confidence threshold for face detection (default: 0.5)")
    parser.add_argument("--face_padding", type=float, default=0.2, 
                        help="Padding around detected faces (default: 0.2)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    if args.video:
        extract_faces_from_video(
            args.video, 
            args.output, 
            args.yolo, 
            args.sample_rate, 
            args.frames_per_batch,
            args.face_conf, 
            args.face_padding
        )
    elif args.video_dir:
        process_videos_in_directory(
            args.video_dir, 
            args.output, 
            args.yolo, 
            args.sample_rate, 
            args.frames_per_batch,
            args.face_conf, 
            args.face_padding
        )
    else:
        print("Please provide either --video or --video_dir argument")