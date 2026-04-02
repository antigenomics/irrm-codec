import torch
import torch.nn as nn

from irrm_codec.tokenization import EOS_ID, PAD_ID


class InverseModel(nn.Module):
    def __init__(
        self,
        vocab_size=24,
        embedding_dim=9000,
        hidden_dim=512,
        max_len=40,
        dropout=0.2,
        num_layers=3,
        nhead=8,
        ff_mult=4,
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, 4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.len_head = nn.Linear(hidden_dim, max_len + 1)

        self.pos_emb = nn.Parameter(torch.randn(max_len + 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out = nn.Linear(hidden_dim, vocab_size)

    def encode_embedding(self, emb):
        if emb.ndim != 2:
            raise ValueError(f"Expected embeddings with shape [batch, dim], got {emb.shape}.")
        return self.proj(emb)

    def forward(self, emb, decoder_input=None):
        z = self.encode_embedding(emb)  # [B, H]

        seq_len = self.max_len + 1  # predict tokens including EOS position
        x = z.unsqueeze(1).expand(-1, seq_len, -1) + self.pos_emb[:seq_len].unsqueeze(0)
        x = self.decoder(x)

        logits = self.out(x)          # [B, max_len + 1, vocab]
        len_logits = self.len_head(z) # [B, max_len + 1]
        return logits, len_logits

    @torch.no_grad()
    def generate(self, emb, max_len=None):
        self.eval()

        logits, len_logits = self.forward(emb)
        predicted_lengths = len_logits.argmax(dim=-1)

        max_decode_len = self.max_len if max_len is None else min(max_len, self.max_len)
        step_logits = logits[:, : max_decode_len + 1]
        raw_tokens = step_logits.argmax(dim=-1)

        batch_size = emb.size(0)
        generated = []

        for i in range(batch_size):
            seq = raw_tokens[i]
            pred_len = int(predicted_lengths[i].item())

            if pred_len < 0:
                pred_len = 0
            if pred_len > max_decode_len:
                pred_len = max_decode_len

            tokens = seq[: pred_len + 1].clone()

            eos_positions = (tokens == EOS_ID).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                eos_pos = int(eos_positions[0].item())
                tokens = tokens[: eos_pos + 1]
            else:
                if tokens.numel() == 0:
                    tokens = torch.tensor([EOS_ID], device=emb.device, dtype=torch.long)
                elif tokens[-1].item() != EOS_ID:
                    tokens[-1] = EOS_ID

            generated.append(tokens)

        padded = torch.full(
            (batch_size, max(t.numel() for t in generated)),
            PAD_ID,
            dtype=torch.long,
            device=emb.device,
        )
        for i, tokens in enumerate(generated):
            padded[i, : tokens.numel()] = tokens

        return padded, predicted_lengths
