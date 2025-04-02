import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# 1. Load and preprocess images
def load_images(folder_path):
    images = []
    file_paths = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            # Resize to a standard size if necessary
            img = img.resize((160, 160))  # Standard size for many face recognition models
            images.append(np.array(img))
            file_paths.append(file_path)
    return np.array(images), file_paths

# 2. Get face embeddings
def get_embeddings(images):
    # Load a model pre-trained on low-resolution faces if available
    # or use a general face recognition model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        batch_tensor = [torch.Tensor(img.transpose((2, 0, 1))).float() for img in batch]
        batch_tensor = torch.stack(batch_tensor)
        # Normalize
        batch_tensor = (batch_tensor - 127.5) / 128.0
        
        with torch.no_grad():
            batch_embeddings = resnet(batch_tensor)
        
        embeddings.extend(batch_embeddings.cpu().numpy())
    
    return np.array(embeddings)

# 3. Cluster embeddings
def cluster_faces(embeddings, eps=0.5, min_samples=5):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    return clustering.labels_

# 4. Organize images by cluster
def organize_by_cluster(file_paths, labels, output_dir):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for each cluster
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # Noise points
            os.makedirs(os.path.join(output_dir, "unknown"), exist_ok=True)
        else:
            os.makedirs(os.path.join(output_dir, f"person_{label}"), exist_ok=True)
    
    # Copy images to respective cluster directories
    for file_path, label in zip(file_paths, labels):
        filename = os.path.basename(file_path)
        if label == -1:
            dest_path = os.path.join(output_dir, "unknown", filename)
        else:
            dest_path = os.path.join(output_dir, f"person_{label}", filename)
        shutil.copy(file_path, dest_path)

# Run the pipeline
def main():
    input_folder = "base_dataset/raw_faces"
    output_folder = "data/classified_faces"
    
    print("Loading images...")
    images, file_paths = load_images(input_folder)
    
    print("Extracting embeddings...")
    embeddings = get_embeddings(images)
    
    print("Clustering faces...")
    # You'll need to experiment with these parameters
    labels = cluster_faces(embeddings, eps=0.4, min_samples=3)
    
    print("Organizing images by identity...")
    organize_by_cluster(file_paths, labels, output_folder)
    
    print(f"Done! Classified into {len(set(labels)) - (1 if -1 in labels else 0)} identities")
    
if __name__ == "__main__":
    main()