import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
import argparse
import dlib  # For facial landmark detection

def normalize_face(face_img, target_size=(128, 128), use_landmarks=True, use_grayscale=False):
    """
    Normalize and enhance a face image, return success flag, processed image, and rejection reason.
    
    Args:
        face_img: The face image to process
        target_size: Output size (default 128x128 for LightCNN)
        use_landmarks: Whether to attempt face alignment using landmarks
        use_grayscale: Whether to convert to grayscale or keep color
    
    Returns:
        (success, normalized_face, reason): Tuple with success flag, processed image and rejection reason
    """
    # Work with original color or convert to grayscale based on parameter
    if use_grayscale and len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        processed = gray
    else:
        processed = face_img.copy()
    
    # If we're keeping color, apply enhancements to each channel
    if len(processed.shape) == 3:
        # Process each channel separately
        enhanced = np.zeros_like(processed)
        for i in range(3):
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced[:,:,i] = clahe.apply(processed[:,:,i])
        
        # Reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(processed)
        
        # Reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Default to no landmark detection success
    landmarks_detected = False
    normalized_face = denoised
    
    # For landmark detection, we need grayscale specifically for dlib
    if use_landmarks:
        try:
            # For landmark detection, convert to grayscale if needed
            if len(denoised.shape) == 3:
                gray_for_detection = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
            else:
                gray_for_detection = denoised
                
            # Initialize dlib's face detector and landmark predictor
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor(predictor_path)
                
                # Detect face and landmarks
                dlib_rects = detector(gray_for_detection, 1)
                if len(dlib_rects) == 0:
                    return False, None, "No face detected by dlib"
                
                dlib_rect = dlib_rects[0]
                landmarks = predictor(gray_for_detection, dlib_rect)
                
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
                
                # Set success flag
                landmarks_detected = True
        except Exception as e:
            return False, None, f"Landmark detection failed: {e}"
    
    # Only proceed if landmarks were detected (if required)
    if use_landmarks and not landmarks_detected:
        return False, None, "No landmarks detected"
    
    # Resize to target size
    normalized_face = cv2.resize(normalized_face, target_size)
    
    return True, normalized_face, "Success"

def detect_and_save_faces(video_path, output_dir="/mnt/data/PROJECTS/face-rec-lightcnn/base_dataset/raw_faces", 
                         confidence=0.8, min_face_size=20, use_grayscale=False):
    """
    Detect faces in a video using YOLOv8 and save only high quality faces.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save cropped faces
        confidence (float): Confidence threshold for face detection
        min_face_size (int): Minimum width/height for a face to be considered
        use_grayscale (bool): Whether to convert to grayscale
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model for face detection
    model = YOLO("weights/yolo11n-face.pt")
    
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
    
    # Counters for detailed report
    frame_count = 0
    faces_detected = 0
    faces_saved = 0
    
    # Rejection counters
    rejected_low_confidence = 0
    rejected_too_small = 0
    rejected_empty_crop = 0
    rejected_no_face_by_dlib = 0 
    rejected_no_landmarks = 0
    rejected_other_reasons = 0
    rejection_reasons = {}
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 15 != 0:  # Process every 30th frame to save compute
            continue
        
        # Detect faces on the original resolution 
        results = model(frame, conf=confidence, imgsz=(original_height, original_width))
        
        # Process detections
        if results[0].boxes.data.shape[0] > 0:
            for i, box in enumerate(results[0].boxes.data):
                x1, y1, x2, y2, conf, _ = box.cpu().numpy()
                faces_detected += 1
                
                if conf < confidence:
                    rejected_low_confidence += 1
                    continue
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get face dimensions
                w, h = x2 - x1, y2 - y1
                
                # Skip if face is too small
                if w < min_face_size or h < min_face_size:
                    rejected_too_small += 1
                    continue
                
                # Add margins (20% of width/height)
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
                    rejected_empty_crop += 1
                    continue
                
                # Apply face normalization and check if landmarks were detected
                success, normalized_face, reason = normalize_face(
                    face, 
                    target_size=(128, 128), 
                    use_landmarks=True,
                    use_grayscale=use_grayscale
                )
                
                # Only save if normalization was successful and landmarks were detected
                if success and normalized_face is not None:
                    # Create unique filename
                    timestamp = int(time.time() * 1000)
                    filename = f"{video_name}_frame{frame_count}_face{i}_{timestamp}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save the normalized face
                    cv2.imwrite(filepath, normalized_face)
                    faces_saved += 1
                    
                    if faces_saved % 10 == 0:
                        print(f"Saved {faces_saved} faces so far...")
                else:
                    # Track specific rejection reasons
                    if reason == "No face detected by dlib":
                        rejected_no_face_by_dlib += 1
                    elif reason == "No landmarks detected":
                        rejected_no_landmarks += 1
                    else:
                        rejected_other_reasons += 1
                        # Track detailed other reasons
                        if reason not in rejection_reasons:
                            rejection_reasons[reason] = 1
                        else:
                            rejection_reasons[reason] += 1
    
    # Release video
    cap.release()
    
    # Print detailed rejection report
    total_rejected = (rejected_low_confidence + rejected_too_small + rejected_empty_crop +
                     rejected_no_face_by_dlib + rejected_no_landmarks + rejected_other_reasons)
    
    print("\n" + "="*50)
    print(f"FACE DETECTION REPORT FOR: {video_path}")
    print("="*50)
    print(f"Processed {frame_count} frames from video")
    print(f"Total faces detected by YOLO: {faces_detected}")
    print(f"Total faces saved: {faces_saved}")
    print(f"Total faces rejected: {total_rejected}")
    print("\nREJECTION BREAKDOWN:")
    print(f"- Low confidence (below {confidence}): {rejected_low_confidence}")
    print(f"- Too small (below {min_face_size}px): {rejected_too_small}")
    print(f"- Empty crop: {rejected_empty_crop}")
    print(f"- No face detected by dlib: {rejected_no_face_by_dlib}")
    print(f"- No landmarks detected: {rejected_no_landmarks}")
    print(f"- Other reasons: {rejected_other_reasons}")
    
    if rejected_other_reasons > 0:
        print("\nDETAILED OTHER REJECTION REASONS:")
        for reason, count in rejection_reasons.items():
            print(f"- {reason}: {count}")
    
    print("="*50)

def process_videos(video_dir, output_dir="/mnt/data/PROJECTS/face-rec-lightcnn/base_dataset/raw_faces", 
                 confidence=0.8, min_face_size=20, use_grayscale=False):
    """
    Process all video files in a directory
    """
    for file in os.listdir(video_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, file)
            print(f"Processing {video_path}...")
            detect_and_save_faces(video_path, output_dir, confidence, min_face_size, use_grayscale)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and crop high-quality faces from videos")
    parser.add_argument("--video", help="Path to a single video file")
    parser.add_argument("--video_dir", help="Directory containing video files")
    parser.add_argument("--output", default="/mnt/data/PROJECTS/face-rec-lightcnn/base_dataset/raw_faces", 
                        help="Output directory for cropped faces")
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence threshold for detection")
    parser.add_argument("--min_size", type=int, default=20, help="Minimum face size in pixels")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale (default: keep color)")
    
    args = parser.parse_args()
    
    if args.video:
        detect_and_save_faces(args.video, args.output, args.conf, args.min_size, args.grayscale)
    elif args.video_dir:
        process_videos(args.video_dir, args.output, args.conf, args.min_size, args.grayscale)
    else:
        print("Please provide either --video or --video_dir argument")