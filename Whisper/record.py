import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import os

fs = 16000  # Sample rate
duration = 5  # seconds per recording
output_dir = "my_audio_dataset"
os.makedirs(output_dir, exist_ok=True)

data = []

for i in range(10):
    input(f"\n🎙️ Press Enter to start recording audio {i+1}/10...")
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording finished.")

    audio_filename = f"audio_{i+1}.wav"
    full_path = os.path.join(output_dir, audio_filename)
    write(full_path, fs, audio)

    transcript = input("📝 Enter your transcript: ")
    data.append({"filename": audio_filename, "transcript": transcript})

# Save metadata
df = pd.DataFrame(data)
df.to_csv(os.path.join(output_dir, "transcripts.csv"), index=False)

print("\n📁 Dataset saved in:", output_dir)
