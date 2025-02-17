import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from LightCNN.light_cnn import LightCNN_29Layers_v2  # Import LightCNN model

# Custom Dataset
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {person: i for i, person in enumerate(os.listdir(root_dir))}

        for person in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person)
            for img_name in os.listdir(person_dir):
                self.image_paths.append(os.path.join(person_dir, img_name))
                self.labels.append(self.class_to_idx[person])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  # Grayscale
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = FaceDataset(root_dir="/mnt/nvme0n1p4/PROJECTS/face-recognition/data/dataset_1_grayscale", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Get number of classes from dataset
num_classes = len(train_dataset.class_to_idx)

# Initialize model with correct number of classes
model = LightCNN_29Layers_v2(num_classes=num_classes)
model = model.to(device)

# Load Pretrained Weights (Checkpoint)
checkpoint_path = "/mnt/nvme0n1p4/PROJECTS/face-recognition/checkpoints/LightCNN_29Layers_V2_checkpoint.pth.tar"

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Remove "module." prefix from keys
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}

        # Load pretrained weights (excluding fc2)
        model.load_state_dict(new_state_dict, strict=False)

        # Manually initialize fc2 layer with correct output size
        model.fc2 = nn.Linear(256, num_classes).to(device)

        print("✅ Loaded pretrained weights & fixed fc2 layer!")

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}. Training from scratch.")
else:
    print("⚠️ Checkpoint file not found. Training from scratch.")

# Loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)[1]  # ✅ Extract only logits
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Save trained model
torch.save(model.state_dict(), "lightcnn_face_recognizer.pth")
