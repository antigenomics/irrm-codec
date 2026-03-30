import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class ForwardModel(nn.Module):
    def __init__(self, vocab_size=24, embedding_dim=64, hidden_dim=256, output_dim=9000):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.proj = nn.Conv1d(embedding_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dilation) for dilation in [1, 1, 2, 2, 4, 8]]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, output_dim),
        )

    def forward(self, tokens, mask):
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [batch, seq], got {tokens.shape}.")

        mask = mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("Encountered an empty sequence in the batch.")

        x = self.emb(tokens).transpose(1, 2)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)

        expanded_mask = mask.unsqueeze(1)
        denom = expanded_mask.sum(-1).clamp_min(1)
        mean_pool = (x * expanded_mask).sum(-1) / denom
        max_pool = x.masked_fill(~expanded_mask, torch.finfo(x.dtype).min).max(-1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.mlp(pooled)
