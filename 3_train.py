import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the model blueprint we just made
# (Make sure 2_model.py is in the same folder!)
from model_2 import HMSANet

# --- 1. Load Data ---
print("Loading data...")
try:
    X = np.load("processed_data/X.npy")
    y = np.load("processed_data/y.npy")
    classes = np.load("processed_data/classes.npy")
except FileNotFoundError:
    print("Error: processed_data not found! Did you run 1_process_data.py?")
    exit()

# Convert numpy arrays to PyTorch Tensors (Math format for AI)
# We add a 'channel' dimension: (N, 128, 130) -> (N, 1, 128, 130)
X = torch.tensor(X).unsqueeze(1).float()
y = torch.tensor(y).long()

# Split: 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create Data Loaders (Feeds data to the AI in small batches)
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# --- 2. Setup Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = HMSANet(num_classes=len(classes)).to(device)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# --- 3. Training Loop ---
EPOCHS = 30  # Number of times to study the full dataset
best_acc = 0.0

print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    # Train
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()           # Reset gradients
        outputs = model(inputs)         # Forward pass (Guess)
        loss = criterion(outputs, labels)  # Calculate error
        loss.backward()                 # Backward pass (Learn)
        optimizer.step()                # Update weights

        running_loss += loss.item()

    # Evaluate (Test)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f} | Accuracy = {accuracy:.2f}%")

    # Save the best model
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "best_beatbrain_model.pth")

print(f"\nTraining Complete!")
print(f"Best Accuracy Achieved: {best_acc:.2f}%")
print("Model saved as 'best_beatbrain_model.pth'")
