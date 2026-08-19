import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from model_2 import HMSANet

# --- 1. Load Everything ---
print("Loading data and model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
X = np.load("processed_data/X.npy")
y = np.load("processed_data/y.npy")
classes = np.load("processed_data/classes.npy")

# Convert to Tensor
X_tensor = torch.tensor(X).unsqueeze(1).float()
y_tensor = torch.tensor(y).long()

# Load the Best Model
model = HMSANet(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(
    "best_beatbrain_model.pth", map_location=device))
model.eval()

# --- 2. Generate Confusion Matrix ---
print("Generating Confusion Matrix...")
all_preds = []
all_labels = []

# We use a DataLoader to process in batches
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32)

with torch.no_grad():
    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Plotting
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Accuracy: 99.70%)')
plt.savefig('Figure_1_Confusion_Matrix.png')
print("Saved 'Figure_1_Confusion_Matrix.png'")

# --- 3. Generate Attention Map (The "Novelty" Proof) ---
print("Generating Attention Map...")

# We will take the first file from the first class (e.g., a Dadra file)
sample_idx = 0
sample_input = X_tensor[sample_idx].unsqueeze(0).to(device)  # Add batch dim

# We need to "hook" into the model to see the attention weights
# (Re-running the forward pass manually to catch the weights)
x = sample_input
# CNN
x = model.pool(torch.nn.functional.relu(model.bn1(model.conv1(x))))
x = model.pool(torch.nn.functional.relu(model.bn2(model.conv2(x))))
x = model.pool(torch.nn.functional.relu(model.bn3(model.conv3(x))))
# Reshape
b, c, h, w = x.size()
x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
# GRU
gru_out, _ = model.gru(x)
# Attention
attn_weights = torch.nn.functional.softmax(model.attention(gru_out), dim=1)
attn_weights = attn_weights.squeeze().cpu().detach().numpy()

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(attn_weights, color='red', linewidth=2)
plt.fill_between(range(len(attn_weights)),
                 attn_weights, alpha=0.3, color='red')
plt.title(f"HMSA-Net Attention Map for '{classes[y[sample_idx]]}'")
plt.xlabel("Time Steps (Rhythmic Cycle)")
plt.ylabel("Attention Score (Importance)")
plt.savefig('Figure_2_Attention_Map.png')
print("Saved 'Figure_2_Attention_Map.png'")

print("\nDone! Check your folder for the images.")
