import os
import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.module2_classical.features import FeatureExtractor
from src.module2_classical.classifiers import ClassicalClassifier
from src.module3_deep.model import TrafficSignNet

# Configurations
DATASET_PATH = "dataset"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='Train', limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if split == 'Train':
            train_path = os.path.join(root_dir, 'Train')
            for class_id in os.listdir(train_path):
                class_path = os.path.join(train_path, class_id)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.endswith('.png'):
                            self.samples.append((os.path.join(class_path, img_name), int(class_id)))
        
        if limit and len(self.samples) > limit:
            # Shuffle and limit
            np.random.shuffle(self.samples)
            self.samples = self.samples[:limit]
            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def train_classical():
    print("Training Classical Model (HOG + SVM)...")
    # Limit data for speed in this demo
    dataset = GTSRBDataset(DATASET_PATH, limit=2000) 
    
    extractor = FeatureExtractor(method='hog')
    X = []
    y = []
    
    for i in range(len(dataset)):
        img, label = dataset[i]
        # HOG needs gray
        features, _ = extractor.extract_hog(img)
        X.append(features)
        y.append(label)
        
    clf = ClassicalClassifier(method='svm') # or 'rf' for speed
    clf.train(np.array(X), np.array(y))
    
    with open(os.path.join(MODELS_DIR, 'classical_model.pkl'), 'wb') as f:
        pickle.dump(clf.model, f)
    print("Classical model saved.")

def train_deep():
    print("Training Deep Learning Model...")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = GTSRBDataset(DATASET_PATH, transform=transform, split='Train', limit=5000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = TrafficSignNet(num_classes=43)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = 2 # Keeping it short for demo
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {100 * correct / total:.2f}%")
        
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'deep_model.pth'))
    print("Deep model saved.")

if __name__ == "__main__":
    train_classical()
    train_deep()
