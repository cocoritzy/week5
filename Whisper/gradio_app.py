import gradio as gr
import torch
import whisper
import librosa

# Load base model structure (Whisper tiny)
model = whisper.load_model("tiny")

# Load your fine-tuned weights
model.load_state_dict(torch.load("whisper_finetuned/model.pt"))
model.eval()

def transcribe(audio_path):
    # Load audio file and preprocess
    audio, _ = librosa.load(audio_path, sr=16000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Transcribe using decoding options (no timestamps, force English)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    result = whisper.decode(model, mel, options)
    return result.text

# Gradio interface
gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="🎙️ Record your voice"),
    outputs=gr.Textbox(label="📝 Transcription"),
    title="Whisper Fine-Tuned Transcriber"
).launch()
