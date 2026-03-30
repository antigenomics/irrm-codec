import torch
import torch.nn as nn

from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID


class InverseModel(nn.Module):
    def __init__(
        self,
        vocab_size=24,
        embedding_dim=9000,
        hidden_dim=512,
        token_dim=64,
        max_len=40,
        dropout=0.2,
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
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
        self.emb = nn.Embedding(vocab_size, token_dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(
            token_dim,
            hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def encode_embedding(self, emb):
        if emb.ndim != 2:
            raise ValueError(f"Expected embeddings with shape [batch, dim], got {emb.shape}.")
        return self.proj(emb)

    def forward(self, emb, decoder_input):
        z = self.encode_embedding(emb)
        h0 = z.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        x = self.emb(decoder_input)
        out, _ = self.gru(x, h0)
        logits = self.out(out)
        return logits, self.len_head(z)

    @torch.no_grad()
    def generate(self, emb, max_len=None):
        self.eval()
        max_decode_len = self.max_len if max_len is None else max_len
        z = self.encode_embedding(emb)
        hidden = z.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        predicted_lengths = self.len_head(z).argmax(dim=-1)

        batch_size = emb.size(0)
        current = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=emb.device)
        generated = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=emb.device)

        for _ in range(max_decode_len + 1):
            x = self.emb(current[:, -1:])
            out, hidden = self.gru(x, hidden)
            step_logits = self.out(out[:, -1])
            next_token = step_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, EOS_ID), next_token)
            generated.append(next_token)
            finished = finished | next_token.eq(EOS_ID)
            current = torch.cat([current, next_token.unsqueeze(1)], dim=1)
            if finished.all():
                break

        if generated:
            tokens = torch.stack(generated, dim=1)
        else:
            tokens = torch.empty((batch_size, 0), dtype=torch.long, device=emb.device)

        return tokens, predicted_lengths
