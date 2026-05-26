import torch
import torch.nn as nn
import torch.nn.functional as F

from irrm_codec.tokenization import PAD_ID, encode


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


class ConvBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class ForwardModel(nn.Module):
    def __init__(
        self,
        vocab_size=25,
        embedding_dim=64,
        hidden_dim=192,
        mlp_dim=512,
        mlp_hidden_dim=1024,
        dropout=0.2,
        dilations=(1, 2, 4, 8),
        encoder_type="residual",
        output_dim=9000,
        max_len=40,
    ):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.proj = nn.Conv1d(embedding_dim, hidden_dim, 1)
        if encoder_type == "residual":
            block_cls = ResidualBlock
        elif encoder_type == "plain_conv":
            block_cls = ConvBlock
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.blocks = nn.ModuleList(
            [block_cls(hidden_dim, dilation, dropout=dropout) for dilation in dilations]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, output_dim),
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

    @torch.no_grad()
    def predict(self, cdr3_list, device=None):
        if isinstance(cdr3_list, str):
            cdr3_list = [cdr3_list]
        else:
            cdr3_list = list(cdr3_list)

        if not cdr3_list:
            raise ValueError("cdr3_list must contain at least one sequence.")

        model_device = next(self.parameters()).device if device is None else torch.device(device)
        token_tensors = [
            torch.tensor(encode(seq, max_len=self.max_len), dtype=torch.long) for seq in cdr3_list
        ]
        tokens = torch.nn.utils.rnn.pad_sequence(
            token_tensors,
            batch_first=True,
            padding_value=PAD_ID,
        ).to(model_device)
        mask = tokens.ne(PAD_ID)

        was_training = self.training
        self.eval()
        pred = self(tokens, mask)
        if was_training:
            self.train()
        return pred
