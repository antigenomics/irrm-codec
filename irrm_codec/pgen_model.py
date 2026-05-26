import torch

from irrm_codec.forward_model import ForwardModel


class PgenModel(ForwardModel):
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
        max_len=40,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            dilations=dilations,
            encoder_type=encoder_type,
            output_dim=1,
            max_len=max_len,
        )

    def forward(self, tokens, mask):
        return super().forward(tokens, mask).squeeze(-1)

    @torch.no_grad()
    def predict(self, cdr3_list, device=None):
        return super().predict(cdr3_list, device=device).squeeze(-1)
