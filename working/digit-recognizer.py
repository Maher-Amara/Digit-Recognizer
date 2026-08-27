import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Force CPU usage for local testing
device = torch.device('cpu')
print(f"Testing locally on device: {device}")

# 2. Data Loading
# Tries to find the CSVs in the parent directory first (typical Kaggle download setup), otherwise local.
train_path = '../train.csv' if os.path.exists('../train.csv') else 'train.csv'
test_path = '../test.csv' if os.path.exists('../test.csv') else 'test.csv'

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Successfully loaded data. Train shape: {train_df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find train.csv or test.csv at '{train_path}'. Please ensure the Kaggle dataset is downloaded.")
    exit(1)

# 3. Dataset Definition
class MNISTDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.is_test = is_test
        self.transform = transform
        
        if not is_test:
            self.labels = df['label'].values
            self.images = df.drop('label', axis=1).values.astype(np.float32)
        else:
            self.images = df.values.astype(np.float32)
            
        # Reshape to 28x28 and normalize
        self.images = self.images.reshape(-1, 28, 28) / 255.0
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        img_tensor = torch.tensor(img).unsqueeze(0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        if not self.is_test:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img_tensor, label
        return img_tensor

# 4. Small Model Architecture for Fast CPU Training
# Uses only 2 blocks and fewer filters (16 -> 32 -> 64)
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Dense Head (28x28 -> pool -> 14x14 -> pool -> 7x7)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Data Splitting
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)

# Simplified dataloading (No heavy augmentation for fast local testing)
train_dataset = MNISTDataset(train_data)
val_dataset = MNISTDataset(val_data)
test_dataset = MNISTDataset(test_df, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model
model = SmallCNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {total_params:,} parameters (Very lightweight!).")

# 5. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop (Reduced to 5 epochs for local testing)
epochs = 5
print(f"\\nStarting training for {epochs} epochs on CPU...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    train_acc = 100 * correct_train / total_train
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    val_acc = 100 * correct_val / total_val
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

# 7. Prediction & Submission
print("\\nGenerating predictions for test set...")
model.eval()
predictions = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})

submission.to_csv('local_submission.csv', index=False)
print("Saved predictions to 'local_submission.csv'")
