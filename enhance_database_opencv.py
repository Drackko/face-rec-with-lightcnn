import os
import argparse
import cv2
import numpy as np

def print_opencv_info():
    """Print OpenCV version and build information for debugging"""
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV build information: {cv2.getBuildInformation()}")

def bicubic_upscale(img, scale=4):
    """Fallback method using bicubic interpolation"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def enhance_database(input_db_dir, output_db_dir, model_path, min_size=None):
    """
    Enhance all face images in a database directory using EDSR or fallback to bicubic.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_db_dir, exist_ok=True)
    
    # Print OpenCV information for debugging
    print_opencv_info()
    
    # Try to initialize the super resolution model
    use_super_res = False
    try:
        print(f"Loading EDSR model from {model_path}...")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", 4)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Test on a small image to make sure it works
        test_img = np.zeros((32, 32, 3), dtype=np.uint8)
        try:
            sr.upsample(test_img)
            use_super_res = True
            print("Super resolution model loaded successfully!")
        except Exception as e:
            print(f"Failed to run super resolution test: {e}")
            print("Falling back to bicubic upscaling")
    except Exception as e:
        print(f"Error initializing super resolution: {e}")
        print("Falling back to bicubic upscaling")
    
    # Get all person folders
    person_folders = [d for d in os.listdir(input_db_dir) 
                     if os.path.isdir(os.path.join(input_db_dir, d))]
    
    print(f"Found {len(person_folders)} person folders in the database")
    
    # Process each person folder
    for i, person in enumerate(person_folders):
        print(f"Processing person {i+1}/{len(person_folders)}: {person}")
        
        # Create input and output paths for this person
        person_input_dir = os.path.join(input_db_dir, person)
        person_output_dir = os.path.join(output_db_dir, person)
        
        # Create output directory for this person
        os.makedirs(person_output_dir, exist_ok=True)
        
        # Process all images in this person's folder
        image_files = [f for f in os.listdir(person_input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"  Found {len(image_files)} images for {person}")
        processed_count = 0
        
        for img_file in image_files:
            input_path = os.path.join(person_input_dir, img_file)
            output_path = os.path.join(person_output_dir, img_file)
            
            # Read image
            img = cv2.imread(input_path)
            if img is None:
                print(f"  Warning: Could not read {input_path}, skipping")
                continue
                
            # Check if image needs processing based on min_size
            if min_size and min(img.shape[0], img.shape[1]) >= min_size:
                # Just copy the image if it's already large enough
                cv2.imwrite(output_path, img)
                processed_count += 1
                continue
            
            # Process image
            try:
                if use_super_res:
                    try:
                        enhanced_img = sr.upsample(img)
                    except Exception as e:
                        print(f"  Super resolution failed on {input_path}, falling back to bicubic: {e}")
                        enhanced_img = bicubic_upscale(img)
                else:
                    enhanced_img = bicubic_upscale(img)
                
                cv2.imwrite(output_path, enhanced_img)
                processed_count += 1
                
                # Print progress
                if processed_count % 5 == 0:
                    print(f"  Processed {processed_count}/{len(image_files)} images...")
                
            except Exception as e:
                print(f"  Error processing {input_path}: {str(e)}")
                # Copy original as fallback
                cv2.imwrite(output_path, img)
                processed_count += 1
    
    print(f"Database enhancement complete. Enhanced database saved to {output_db_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance face database using EDSR super-resolution")
    parser.add_argument("--input", required=True, help="Input database directory containing person folders")
    parser.add_argument("--output", required=True, help="Output directory for enhanced database")
    parser.add_argument("--model", required=True, help="Path to EDSR model file (.pb file)")
    parser.add_argument("--min-size", type=int, default=None, 
                        help="Only enhance images smaller than this size (default: enhance all)")
    
    args = parser.parse_args()
    
    enhance_database(
        args.input, 
        args.output, 
        args.model,
        args.min_size
    )