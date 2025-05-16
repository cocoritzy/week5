
import os
import torch
import whisper
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# Load model and tokenizer
model = whisper.load_model('tiny')
tknsr = whisper.tokenizer.get_tokenizer(multilingual=True)

# Load dataset
df = pd.read_csv("my_audio_dataset/transcripts.csv")
dataset = []
for i, row in df.iterrows():
    audio_path = os.path.join("my_audio_dataset", row['filename'])
    transcript = row['transcript']
    dataset.append({"audio_filepath": audio_path, "text": transcript})

# Define dataset class
class AudioTranscriptDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df[idx]['audio_filepath']
        text = self.df[idx]['text']

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        tokens = [tknsr.sot, tknsr.language_token, tknsr.transcribe, tknsr.no_timestamps]
        tokens += tknsr.encode(text)
        tokens += [tknsr.eot]

        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)

        return mel, input_tokens, target_tokens

# Split train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

dataset_class_train = AudioTranscriptDataset(train_dataset)
dataset_class_test = AudioTranscriptDataset(test_dataset)

train_dataloader = DataLoader(dataset_class_train, batch_size=1, shuffle=False)
test_dataloader = DataLoader(dataset_class_test, batch_size=1, shuffle=False)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
model.train()

for epoch in range(10):
    total_loss = 0
    num_batches = 0
    for mel, input_tokens, target_tokens in train_dataloader:
        mel = mel.to(model.device)
        input_tokens = input_tokens.to(model.device)
        target_tokens = target_tokens.to(model.device)

        logits = model(mel, input_tokens)
        loss = criterion(logits.view(-1, logits.size(-1)), target_tokens.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"📉 Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

# # Evaluation
# model.eval()
# with torch.no_grad():
#     for i, (mel, _, target_tokens) in enumerate(test_dataloader):
#         mel = mel.to(model.device)
#         target_tokens = target_tokens.squeeze(0)

#         tokens = [tknsr.sot, tknsr.language_token, tknsr.transcribe, tknsr.no_timestamps]
#         tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(model.device)

#         generated = tokens.clone()
#         max_tokens = 64
#         for _ in range(max_tokens):
#             logits = model(mel, generated)
#             next_token_logits = logits[:, -1, :]
#             next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
#             generated = torch.cat([generated, next_token], dim=1)
#             if next_token.item() == tknsr.eot:
#                 break

#         decoded_pred = tknsr.decode(generated.squeeze(0).tolist())
#         decoded_target = tknsr.decode(target_tokens.tolist())
#         print(f"🧪 Sample {i+1}")
#         print("📜 Generated:", decoded_pred)
#         print("🎯 Target   :", decoded_target)
#         print("-------")

# ✅ Save fine-tuned model
save_dir = "whisper_finetuned"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "model_soir.pt"))