import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar

# --- SETTINGS ---
# This points to: BEATBRAIN/archive/tablaDataset
DATA_PATH = "archive/tablaDataset"
OUTPUT_PATH = "processed_data"
SAMPLE_RATE = 22050
DURATION = 3.0  # We cut all audio to exactly 3 seconds

# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)


def extract_features(y, sr):
    # Convert audio wave to an image (Spectrogram)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    return melspec


def augment_audio(y, sr):
    # This creates "fake" new examples to help the AI learn better

    # 1. Time Stretch (Speed up/Slow down slightly)
    rate = np.random.uniform(0.8, 1.2)
    y_stretch = librosa.effects.time_stretch(y, rate=rate)

    # 2. Pitch Shift (Make it slightly higher/lower pitched)
    steps = np.random.randint(-2, 2)
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    return [y_stretch, y_shift]


data = []
labels = []

# These are the folders inside tablaDataset (dadra, teental, etc.)
classes = [d for d in os.listdir(
    DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
# Remove 'wavfiles' if it was accidentally picked up
if 'wavfiles' in classes:
    classes.remove('wavfiles')

print(f"Found Taals: {classes}")

# Loop through every folder
for label_idx, taal_name in enumerate(classes):
    folder_dir = os.path.join(DATA_PATH, taal_name)
    print(f"Processing {taal_name}...")

    # Check every file in the folder
    for file_name in tqdm(os.listdir(folder_dir)):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_dir, file_name)

            try:
                # Load Audio
                y, sr = librosa.load(
                    file_path, sr=SAMPLE_RATE, duration=DURATION)

                # Force audio to be exactly 3 seconds long
                target_len = int(SAMPLE_RATE * DURATION)
                if len(y) < target_len:
                    # Add silence if too short
                    y = np.pad(y, (0, target_len - len(y)))
                else:
                    y = y[:target_len]  # Cut if too long

                # 1. Save Original
                feat = extract_features(y, sr)
                data.append(feat)
                labels.append(label_idx)

                # 2. Save Augmented Versions (Triples your dataset size!)
                aug_versions = augment_audio(y, sr)
                for aug_y in aug_versions:
                    # Fix length for augmented versions too
                    if len(aug_y) < target_len:
                        aug_y = np.pad(aug_y, (0, target_len - len(aug_y)))
                    else:
                        aug_y = aug_y[:target_len]

                    feat_aug = extract_features(aug_y, sr)
                    data.append(feat_aug)
                    labels.append(label_idx)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Save the math data to files
print("Saving data... this might take a minute.")
np.save(f"{OUTPUT_PATH}/X.npy", np.array(data))
np.save(f"{OUTPUT_PATH}/y.npy", np.array(labels))
np.save(f"{OUTPUT_PATH}/classes.npy", classes)
print("Done! Data saved to 'processed_data/' folder.")
