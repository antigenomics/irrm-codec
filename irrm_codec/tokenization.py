AA_VOCAB = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
    "A": 4,
    "C": 5,
    "D": 6,
    "E": 7,
    "F": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "K": 12,
    "L": 13,
    "M": 14,
    "N": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "V": 21,
    "W": 22,
    "Y": 23,
}

ID2AA = {value: key for key, value in AA_VOCAB.items()}

PAD_ID = AA_VOCAB["<PAD>"]
BOS_ID = AA_VOCAB["<BOS>"]
EOS_ID = AA_VOCAB["<EOS>"]
UNK_ID = AA_VOCAB["<UNK>"]


def normalize_sequence(seq):
    if seq is None:
        raise ValueError("Sequence must not be None.")

    seq = str(seq).strip().upper()
    if not seq:
        raise ValueError("Sequence must not be empty.")
    return seq


def encode(seq, max_len=40):
    normalized = normalize_sequence(seq)
    tokens = [AA_VOCAB.get(char, UNK_ID) for char in normalized]
    return tokens[:max_len]


def decode(tokens, stop_at_eos=True):
    decoded = []
    for token in tokens:
        if token == EOS_ID and stop_at_eos:
            break
        if token <= UNK_ID:
            continue
        decoded.append(ID2AA[token])
    return "".join(decoded)
