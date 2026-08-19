

import streamlit as st
import torch
import numpy as np
import librosa
import librosa.display  
import os
import tempfile
import time  # <--- Added this to handle the delay
import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import seaborn as sns
from model_2 import HMSANet

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BeatBrain: Tabla Classifier",
    page_icon="🥁",
    layout="centered"
)

# --- SETTINGS ---
MODEL_PATH = "best_beatbrain_model.pth"
CLASSES_PATH = "processed_data/classes.npy"
SAMPLE_RATE = 22050
DURATION = 3.0

# --- TITLE & DESCRIPTION ---
st.title("🥁 BeatBrain")
st.markdown("### Indian Classical Tabla Taal Recognizer")
st.write("Upload a .wav file, and the AI will identify the Taal (Rhythm).")

# --- LOAD RESOURCES (Cached for Speed) ---


@st.cache_resource
def load_resources():
    # 1. Load Classes
    if not os.path.exists(CLASSES_PATH):
        st.error("❌ Error: processed_data/classes.npy not found!")
        return None, None, None
    classes = np.load(CLASSES_PATH)

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HMSANet(num_classes=len(classes)).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        st.error(f"❌ Error: {MODEL_PATH} not found!")
        return None, None, None

    return model, classes, device


model, classes, device = load_resources()

# --- PROCESSING FUNCTION ---


def process_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Fix length to exactly 3 seconds
    target_len = int(SAMPLE_RATE * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))  # Pad silence
    else:
        y = y[:target_len]  # Cut

    # Create Spectrogram
    melspec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
    melspec = librosa.power_to_db(melspec, ref=np.max)

    return melspec


# --- UI LOGIC ---
uploaded_file = st.file_uploader("Choose a WAV file...", type=["wav"])

if uploaded_file is not None:
    # 1. Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 2. Audio Player Placeholder
    # We use a placeholder so we can update it later
    audio_box = st.empty()

    # Initially show the standard player (Paused)
    audio_box.audio(uploaded_file, format='audio/wav')

    # 3. Analyze Button
    if st.button("🔍 Analyze Rhythm"):

        # STEP A: Start Playing Audio
        audio_box.audio(uploaded_file, format='audio/wav', autoplay=True)

        # STEP B: Wait for 1.5 seconds (Let the user listen)
        time.sleep(1.5)

        with st.spinner("Listening to the beats..."):
            try:
                # STEP C: Perform Analysis
                spectrogram = process_audio(tmp_path)

                # Prepare for Model
                tensor = torch.tensor(spectrogram).float(
                ).unsqueeze(0).unsqueeze(0).to(device)

                # Predict
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probs, 1)

                # Get Result
                predicted_class = classes[predicted_idx.item()]
                conf_score = confidence.item() * 100

                # STEP D: Stop Audio
                # Refresh the player (Pause it) to signal completion
                audio_box.empty()
                audio_box.audio(uploaded_file, format='audio/wav')

                # --- DISPLAY RESULTS ---
                st.success(f"**Prediction:** {predicted_class.upper()}")
                st.metric(label="Confidence", value=f"{conf_score:.2f}%")

                # Probability Bar Chart
                st.subheader("Confidence Breakdown")
                probs_np = probs.cpu().numpy()[0] * 100

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=classes, y=probs_np, palette="viridis", ax=ax)
                ax.set_ylabel("Probability (%)")
                ax.set_ylim(0, 100)
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # Show Spectrogram
                st.subheader("Visual Analysis (Spectrogram)")
                fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
                librosa.display.specshow(
                    spectrogram, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=ax_spec)
                plt.colorbar(ax_spec.collections[0], format='%+2.0f dB')
                st.pyplot(fig_spec)

            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
