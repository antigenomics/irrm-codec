AA_VOCAB = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "-": 4,
    "A": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "V": 22,
    "W": 23,
    "Y": 24,
}

ID2AA = {value: key for key, value in AA_VOCAB.items()}

PAD_ID = AA_VOCAB["<PAD>"]
BOS_ID = AA_VOCAB["<BOS>"]
EOS_ID = AA_VOCAB["<EOS>"]
UNK_ID = AA_VOCAB["<UNK>"]
GAP_TOKEN = "-"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def normalize_sequence(seq):
    if seq is None:
        raise ValueError("Sequence must not be None.")

    seq = str(seq).strip().upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    return seq


def gap_pad_cdr3(seq, target_len=40, left_anchor=4, right_anchor=3):
    normalized = normalize_sequence(seq)
    if any(char not in VALID_AA for char in normalized):
        invalid = sorted({char for char in normalized if char not in VALID_AA})
        raise ValueError(f"Sequence contains unsupported characters before gap insertion: {invalid}")

    if len(normalized) > target_len:
        raise ValueError(
            f"Sequence length {len(normalized)} exceeds target length {target_len} before gap insertion."
        )

    total_gaps = target_len - len(normalized)
    left_gaps = total_gaps // 2
    right_gaps = total_gaps - left_gaps

    left_anchor = min(left_anchor, len(normalized))
    right_start = max(left_anchor, len(normalized) - right_anchor)

    left = normalized[:left_anchor]
    middle = normalized[left_anchor:right_start]
    right = normalized[right_start:]
    return f"{left}{GAP_TOKEN * left_gaps}{middle}{GAP_TOKEN * right_gaps}{right}"


def encode(seq, max_len=40):
    normalized = gap_pad_cdr3(seq, target_len=max_len)
    tokens = [AA_VOCAB.get(char, UNK_ID) for char in normalized]
    return tokens[:max_len]


def strip_gaps(seq):
    normalized = normalize_sequence(seq)
    return normalized.replace(GAP_TOKEN, "")


def decode(tokens, stop_at_eos=True, remove_gaps=False):
    decoded = []
    for token in tokens:
        if token == EOS_ID and stop_at_eos:
            break
        if token <= UNK_ID:
            continue
        decoded.append(ID2AA[token])
    sequence = "".join(decoded)
    if remove_gaps:
        return strip_gaps(sequence)
    return sequence
