import torch
import torch.nn.functional as F

from irrm_codec.tokenization import AA_VOCAB, PAD_ID


def forward_loss(pred, target):
    mse = F.mse_loss(pred, target)
    cos = 1 - F.cosine_similarity(pred, target, dim=-1).mean()
    return 0.7 * mse + 0.3 * cos


def inverse_loss(logits, target):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=PAD_ID,
    )


def pgen_loss(pred, target):
    return F.huber_loss(pred, target, delta=0.5)


def forward_metrics(pred, target):
    mse = F.mse_loss(pred, target).item()
    cosine = F.cosine_similarity(pred, target, dim=-1).mean().item()
    return {"mse": mse, "cosine": cosine}


def inverse_metrics(logits, target):
    with torch.no_grad():
        token_pred = logits.argmax(dim=-1)
        valid_mask = target.ne(PAD_ID)
        token_accuracy = token_pred.eq(target).logical_and(valid_mask).sum().float()
        token_accuracy = (token_accuracy / valid_mask.sum().clamp_min(1)).item()
        predicted_lengths = token_lengths_without_gaps(token_pred)
        target_lengths = token_lengths_without_gaps(target)
        length_accuracy = predicted_lengths.eq(target_lengths).float().mean().item()

    return {
        "token_accuracy": token_accuracy,
        "length_accuracy": length_accuracy,
    }


def pgen_metrics(pred, target):
    with torch.no_grad():
        diff = pred - target
        mse = diff.pow(2).mean().item()
        rmse = mse ** 0.5
        mae = diff.abs().mean().item()
    return {"mse": mse, "rmse": rmse, "mae": mae}


def token_lengths_without_gaps(tokens):
    gap_id = AA_VOCAB["-"]
    return tokens.ne(gap_id).logical_and(tokens.gt(gap_id)).sum(dim=-1)
