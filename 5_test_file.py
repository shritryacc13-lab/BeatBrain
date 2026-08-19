import torch
import numpy as np
import librosa
import os
import sys
from model_2 import HMSANet  # Import your model structure

# --- CONFIGURATION ---
MODEL_PATH = "best_beatbrain_model.pth"
CLASSES_PATH = "processed_data/classes.npy"
SAMPLE_RATE = 22050
DURATION = 3.0  # Seconds


def predict_single_file(file_path):
    # 1. Load the Class Names
    if not os.path.exists(CLASSES_PATH):
        print("Error: processed_data/classes.npy not found.")
        return

    classes = np.load(CLASSES_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Classes known to model: {classes}")

    # 2. Load the Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    # Initialize model with correct number of classes
    model = HMSANet(num_classes=len(classes)).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError:
        print("\n❌ Error: Model shape mismatch!")
        print("Did you re-train after adding Bhajani? You must run 3_train.py first.")
        return

    model.eval()

    # 3. Process the Audio File
    print(f"\nAnalyzing: {file_path}...")
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Fix length to exactly 3 seconds (same as training)
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))  # Pad with silence
        else:
            y = y[:target_len]  # Cut if too long

        # Convert to Spectrogram (Image)
        melspec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=128)
        melspec = librosa.power_to_db(melspec, ref=np.max)

        # Prepare for Model (Add Batch & Channel dimensions)
        # Shape becomes: [1, 1, 128, 130]
        tensor = torch.tensor(melspec).float().unsqueeze(
            0).unsqueeze(0).to(device)

        # 4. Predict
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

            predicted_class = classes[predicted_idx.item()]
            score = confidence.item() * 100

            print(f"🥁 Prediction: {predicted_class.upper()}")
            print(f"📊 Confidence: {score:.2f}%")

            # Show all probabilities (useful for debugging confusion)
            print("\n--- Full Breakdown ---")
            for i, class_name in enumerate(classes):
                prob = probs[0][i].item() * 100
                print(f"{class_name}: {prob:.1f}%")

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    # CHANGE THIS PATH to the file you want to test
    # Example: "archive/tablaDataset/bhajani/some_file.wav"
    # test_file = "archive/tablaDataset/bhajani/bhajani01.wav"
    test_file = "archive/tablaDataset/addhatrital/addhatrital08.wav"

    # Check if user provided a file in command line
    if len(sys.argv) > 1:
        test_file = sys.argv[1]

    if os.path.exists(test_file):
        predict_single_file(test_file)
    else:
        print(f"❌ File not found: {test_file}")
        print("Please edit the 'test_file' variable in the script or provide a path.")
