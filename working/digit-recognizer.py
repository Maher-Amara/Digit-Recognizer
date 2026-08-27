import os
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Epochs: {EPOCHS}")

# ==========================================
# DATA LOADING
# ==========================================
if os.path.exists('../input/competitions/digit-recognizer/train.csv'):
    train_path = '../input/competitions/digit-recognizer/train.csv'
    test_path = '../input/competitions/digit-recognizer/test.csv'
elif os.path.exists('/kaggle/input/digit-recognizer/train.csv'):
    train_path = '/kaggle/input/digit-recognizer/train.csv'
    test_path = '/kaggle/input/digit-recognizer/test.csv'
else:
    train_path = '../train.csv' if os.path.exists('../train.csv') else 'train.csv'
    test_path = '../test.csv' if os.path.exists('../test.csv') else 'test.csv'

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find data files. Checked: {train_path}")
    exit(1)

# ==========================================
# DATASET WITH AUGMENTATION
# ==========================================
class MNISTDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.is_test = is_test
        self.transform = transform
        if not is_test:
            self.labels = df['label'].values
            self.images = df.drop('label', axis=1).values.astype(np.float32)
        else:
            self.images = df.values.astype(np.float32)
        self.images = self.images.reshape(-1, 28, 28) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_tensor = torch.tensor(self.images[idx]).unsqueeze(0)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if not self.is_test:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img_tensor, label
        return img_tensor

# Careful augmentation: small rotation and shift, strictly no flips
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)

train_loader = DataLoader(MNISTDataset(train_data, transform=train_transforms), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MNISTDataset(val_data), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(MNISTDataset(test_df, is_test=True), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# MODEL VARIANTS
# ==========================================



class RecommendedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, padding=1), nn.Dropout2d(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = RecommendedCNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")

# ==========================================
# TRAINING
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

best_val_acc = 0.0
print(f"\nStarting training...")

for epoch in range(EPOCHS):
    start_time = time.time()

    # Train
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0
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

    # Eval
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = 100 * correct_val / total_val
    epoch_time = time.time() - start_time

    # LR Scheduler
    scheduler.step(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        marker = " << BEST"
    else:
        marker = ""

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{EPOCHS}] - {epoch_time:.1f}s | "
          f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f}{marker}")

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

# ==========================================
# PREDICTION (using best saved model)
# ==========================================
print("Loading best model for predictions...")
model.load_state_dict(torch.load('best_model.pth', weights_only=True))
model.eval()

predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

submission = pd.DataFrame({'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission.to_csv('submission.csv', index=False)
print(f"Saved submission.csv ({len(predictions)} predictions)")
