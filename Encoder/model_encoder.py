import torch 
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_length):
        super(CNN1D, self).__init__()
        self.conv1 =nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.gelu = nn.GELU()
    
    
    def forward(self,x):
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))

        # Return for Transformer: (B, T, 256)
        x = x.permute(0, 2, 1)          # → (B, T, 256) #is a dimension swap 

        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, mlp_dimension=512):
        super(EncoderBlock, self).__init__()

        # LayerNorm before attention (PreNorm)
        self.ln1 = nn.LayerNorm(embed_dim)

        # Multihead Self-Attention
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Feed-forward network (MLP block)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dimension),
            nn.ReLU(),
            nn.Linear(mlp_dimension, embed_dim)
        )

        # LayerNorm after MLP
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention block with residual connection
        x_norm = self.ln1(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_output  # Residual connection

        # Feedforward block with residual
        x_norm = self.ln2(x)
        x = x + self.ffn(x_norm)

        return x


class classification(nn.Module):
    def __init__(self, input_length, embed_dim=256, num_heads=4, mlp_dimension=512, num_classes=10, num_layers=1):
        super(classification, self).__init__()
        self.cnn = CNN1D(input_length)  # CNN1D for feature extraction
        self.positional = nn.Parameter(torch.randn(1, 128, embed_dim))  # Positional encoding
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_dimension=mlp_dimension)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)  # (B, 128, T) → (B, T, 256)
        x = x + self.positional[:, :x.size(1)]

        
        for block in self.encoder_blocks:
            x = block(x)  # [B, T, D]

        # ⬇️ Mean pooling over time (dim=1)
        x = x.mean(dim=1)  # (B, 256)
        logits = self.classifier(x)  # [B, num_classes]

        return logits  # Probabilities over classes

