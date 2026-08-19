import torch
import torch.nn as nn
import torch.nn.functional as F


class HMSANet(nn.Module):
    def __init__(self, num_classes):
        super(HMSANet, self).__init__()

        # --- 1. Convolutional Layers (The "Eye") ---
        # These layers look at the Spectrogram image to find patterns
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # --- 2. Bi-Directional GRU (The "Ear") ---
        # This part listens to the rhythm over time (Forward and Backward)
        # Input size calc: 128 channels * (128 mel-bins / 8 pooling reduction) = 128*16
        self.gru = nn.GRU(input_size=128 * 16, hidden_size=64,
                          num_layers=2, batch_first=True, bidirectional=True)

        # --- 3. Attention Mechanism (The "Focus") ---
        # This helps the model decide which beat is the most important (the "Sam")
        # 128 because 64 hidden * 2 directions
        self.attention = nn.Linear(128, 1)

        # --- 4. Classifier ---
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)  # Prevents memorizing (overfitting)

    def forward(self, x):
        # x shape: [Batch, 1, 128, 130] (1 Channel, Freq, Time)

        # Pass through CNN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Prepare for GRU (Reshape)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)

        # Pass through GRU
        gru_out, _ = self.gru(x)

        # Calculate Attention
        attn_weights = F.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)

        # Final Prediction
        out = self.dropout(context_vector)
        out = self.fc(out)
        return out
