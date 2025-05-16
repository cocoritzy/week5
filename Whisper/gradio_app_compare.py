import gradio as gr
import torch
import whisper
import librosa

# Load original base model
base_model = whisper.load_model("tiny")

# Load fine-tuned model (same structure, new weights)
finetuned_model = whisper.load_model("tiny")
finetuned_model.load_state_dict(torch.load("whisper_finetuned/model_soir.pt"))
finetuned_model.eval()

def compare_transcriptions(audio_path):
    # Load and preprocess audio
    audio, _ = librosa.load(audio_path, sr=16000)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(base_model.device)

    # Decoding options
    options = whisper.DecodingOptions(language="en", without_timestamps=True)

    # Transcribe with base model
    result_base = whisper.decode(base_model, mel, options)
    
    # Transcribe with fine-tuned model
    result_finetuned = whisper.decode(finetuned_model, mel, options)

    return result_base.text, result_finetuned.text

# Gradio interface
gr.Interface(
    fn=compare_transcriptions,
    inputs=gr.Audio(type="filepath", label="🎙️ Record or upload your voice"),
    outputs=[
        gr.Textbox(label="🔸 Base model transcription"),
        gr.Textbox(label="🔹 Fine-tuned model transcription")
    ],
    title="🧠 Whisper Comparison: Base vs. Fine-Tuned",
    description="Upload or record an audio file and compare the original Whisper transcription with your fine-tuned model."
).launch()
