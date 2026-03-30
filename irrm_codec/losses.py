import torch
import torch.nn.functional as F

from irrm_codec.tokenization import PAD_ID


def forward_loss(pred, target):
    mse = F.mse_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return 0.7 * mse + 0.3 * cos


def inverse_loss(logits, target, length_logits, lengths):
    seq_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=PAD_ID,
    )
    length_targets = lengths.clamp(min=0, max=length_logits.size(-1) - 1)
    len_loss = F.cross_entropy(length_logits, length_targets)
    return seq_loss + 0.2 * len_loss


def forward_metrics(pred, target):
    mse = F.mse_loss(pred, target).item()
    cosine = F.cosine_similarity(pred, target, dim=-1).mean().item()
    return {"mse": mse, "cosine": cosine}


def inverse_metrics(logits, target, length_logits, lengths):
    with torch.no_grad():
        token_pred = logits.argmax(dim=-1)
        valid_mask = target.ne(PAD_ID)
        token_accuracy = token_pred.eq(target).logical_and(valid_mask).sum().float()
        token_accuracy = (token_accuracy / valid_mask.sum().clamp_min(1)).item()

        length_pred = length_logits.argmax(dim=-1)
        length_accuracy = length_pred.eq(lengths).float().mean().item()

    return {
        "token_accuracy": token_accuracy,
        "length_accuracy": length_accuracy,
    }
