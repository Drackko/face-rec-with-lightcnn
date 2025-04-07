import cv2
import numpy as np
import os

def enhance_image(img, model_path, output_path=None, scale=4):
    """
    Enhance an image using EDSR super-resolution model
    
    Args:
        img (np.ndarray): Input OpenCV image array
        model_path (str): Path to EDSR model (.pb file)
        output_path (str, optional): Path to save output image. If None, image is not saved.
        scale (int): Scale factor - should match model (default: 4)
    
    Returns:
        np.ndarray: Enhanced image (4x scaled)
    """
    # Validate input image
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Input must be a valid OpenCV image array")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Handle grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Handle RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Initialize super resolution model
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", scale)
        
        # Use CPU for reliable processing
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Process image with EDSR
        enhanced_img = sr.upsample(img)
        
    except Exception as e:
        print(f"Super resolution failed, falling back to bicubic: {e}")
        # Fallback to bicubic upscaling
        h, w = img.shape[:2]
        enhanced_img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    
    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, enhanced_img)
    
    return enhanced_img

# Example usage
if __name__ == "__main__":
    # Example of using the function with an OpenCV image
    img = cv2.imread("test.jpg")
    enhanced = enhance_image(img, "EDSR_x4.pb")
    cv2.imwrite("enhanced.jpg", enhanced)