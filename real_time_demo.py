import torch
import numpy as np
import pyaudio
import librosa
import collections
import time
import os
import sys
import threading
from datetime import datetime

# Import your model architecture
from model_2 import HMSANet

# --- CONFIGURATION ---
MODEL_PATH = "best_beatbrain_model.pth"
CLASSES_PATH = "processed_data/classes.npy"
SAMPLE_RATE = 22050
DURATION = 3.0          # Model expects 3 seconds
PREDICTION_INTERVAL = 0.5  # Predict every 0.5 seconds
CHUNK = 2048            # Larger chunk for better audio capture
TIMEOUT = 120           # Stop automatically after 120 seconds
# Voting window for stability (reduced for faster prediction)
SMOOTHING_WINDOW = 3
CONFIDENCE_THRESHOLD = 30  # Lower threshold for more detections
NOISE_GATE = 0.005      # Silence threshold for noise gating
LOG_RESULTS = True      # Save predictions to file


class RealTimeTaal:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎯 Running on: {self.device}")

        # 1. Load Classes & Verify
        try:
            self.classes = np.load(CLASSES_PATH)
            print(f"✅ Classes loaded: {self.classes}")
        except FileNotFoundError:
            print("❌ Error: classes.npy not found. Run 1_process_data.py first.")
            sys.exit(1)

        # 2. Load Model
        self.model = HMSANet(num_classes=len(self.classes)).to(self.device)
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        except FileNotFoundError:
            print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
            sys.exit(1)
        except RuntimeError as e:
            print(f"❌ Model Size Mismatch: {e}")
            sys.exit(1)

        self.model.eval()
        print("✅ Model loaded successfully!")

        # 3. Setup Buffers
        self.buffer_len = int(SAMPLE_RATE * DURATION)
        self.audio_buffer = collections.deque(maxlen=self.buffer_len)
        self.audio_buffer.extend([0.0] * self.buffer_len)

        # Prediction history for smoothing
        self.history = collections.deque(maxlen=SMOOTHING_WINDOW)
        self.predictions_log = []

        # Threading control
        self.stop_event = threading.Event()
        self.last_prediction_time = 0

    def preprocess_audio(self, audio_data):
        """Convert audio to mel-spectrogram - EXACT MATCH to training preprocessing."""
        y = np.array(audio_data).astype(np.float32)

        # Ensure exactly 3 seconds at 22050 Hz = 66150 samples
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Check if all silence
        if np.max(np.abs(y)) < 0.001:
            print("\r⏳ No sound detected...                                         ",
                  end="", flush=True)
            return None

        # Mel-spectrogram - EXACT MATCH to training
        melspec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=128)
        melspec = librosa.power_to_db(melspec, ref=np.max)

        # Convert to tensor - ADD BATCH AND CHANNEL DIMS like training
        # Shape should be: (1, 1, 128, 130) for input to model
        tensor = torch.tensor(melspec).float().unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def start_listening(self):
        """Main audio capture loop with threaded predictions."""
        p = pyaudio.PyAudio()

        # List available audio devices (helpful for debugging)
        print(f"\n📱 Using default input device...")

        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=None,  # Use default device
                start=False
            )
            stream.start_stream()
        except Exception as e:
            print(f"❌ Audio Error: {e}")
            print("   - Check if microphone is connected")
            print("   - Try: python -m sounddevice")
            p.terminate()
            return

        print(f"\n🎤 Listening for {TIMEOUT} seconds...")
        print("-" * 70)

        # Start prediction thread
        prediction_thread = threading.Thread(
            target=self.prediction_loop, daemon=True)
        prediction_thread.start()
        self.last_prediction_time = time.time()

        start_time = time.time()
        frames_read = 0

        try:
            while (time.time() - start_time) < TIMEOUT and not self.stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.float32)
                    self.audio_buffer.extend(samples)
                    frames_read += len(samples)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user.")
        finally:
            self.stop_event.set()
            prediction_thread.join(timeout=2)

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Print summary
            print("\n" + "-" * 70)
            print(f"📊 Session Summary:")
            print(f"   - Duration: {time.time() - start_time:.1f}s")
            print(f"   - Frames captured: {frames_read}")
            print(f"   - Predictions made: {len(self.predictions_log)}")

            if self.predictions_log:
                print(f"\n📈 Prediction Log:")
                # Show last 10
                for i, pred in enumerate(self.predictions_log[-10:], 1):
                    print(
                        f"   {i}. {pred['class'].upper():12s} - {pred['confidence']:5.1f}% ({pred['votes']}/{SMOOTHING_WINDOW} votes)")

            # Save to file
            if LOG_RESULTS and self.predictions_log:
                self.save_results()

    def predict_smoothed(self):
        """Make prediction with voting-based smoothing."""
        input_tensor = self.preprocess_audio(self.audio_buffer)

        if input_tensor is None:
            print("\r⏳ Silence detected... Listening...                           ",
                  end="", flush=True)
            return  # Silence detected

        with torch.no_grad():
            try:
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probs, 1)

                current_class = self.classes[predicted_idx.item()]
                current_conf = confidence.item() * 100

                # Add to history for voting
                self.history.append(current_class)

                # Start showing predictions immediately
                if len(self.history) >= 1:
                    vote_result = collections.Counter(
                        self.history).most_common(1)[0]
                    smoothed_class = vote_result[0]
                    vote_count = vote_result[1]

                    # Always show (even low confidence)
                    print(
                        f"\r🥁 [{smoothed_class.upper():12s}] | {vote_count}/{len(self.history)} votes | {current_conf:5.1f}%    ",
                        end="", flush=True)

                    # Log all predictions
                    self.predictions_log.append({
                        'time': datetime.now().isoformat(),
                        'class': smoothed_class,
                        'confidence': current_conf,
                        'votes': vote_count
                    })
            except Exception as e:
                print(f"\n❌ Error in prediction: {e}", flush=True)
                import traceback
                traceback.print_exc()

    def prediction_loop(self):
        """Separate thread for periodic predictions."""
        while not self.stop_event.is_set():
            current_time = time.time()
            if current_time - self.last_prediction_time >= PREDICTION_INTERVAL:
                self.predict_smoothed()
                self.last_prediction_time = current_time
            time.sleep(0.05)  # Small sleep to avoid busy-waiting

    def save_results(self):
        """Save prediction results to CSV for analysis."""
        import csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.csv"

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['time', 'class', 'confidence', 'votes'])
                writer.writeheader()
                writer.writerows(self.predictions_log)
            print(f"   ✅ Results saved to {filename}")
        except Exception as e:
            print(f"   ⚠️  Could not save results: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("🎵 BeatBrain - Real-Time Tabla Classification")
    print("=" * 70)
    app = RealTimeTaal()
    app.start_listening()
