import torch

from src.config import PAD_IDX


def create_padding_mask(tokens: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """Mask out <pad> positions.

    Args:
        tokens: integer token ids, shape (batch, seq_len).
        pad_idx: index of the padding token.

    Returns:
        Bool tensor of shape (batch, 1, 1, seq_len). The singleton dims
        broadcast across heads and query positions.
    """
    mask = tokens != pad_idx
    return mask.unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Mask out future positions so each token only attends to itself and earlier ones.

    Args:
        seq_len: length of the target sequence.
        device: device to build the mask on, matching the model's tensors.

    Returns:
        Bool lower-triangular tensor of shape (1, 1, seq_len, seq_len).
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tokens: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """Combined padding + causal mask for decoder self-attention.

    Args:
        tokens: target token ids, shape (batch, seq_len).
        pad_idx: index of the padding token.

    Returns:
        Bool tensor of shape (batch, 1, seq_len, seq_len).
    """
    padding = create_padding_mask(tokens, pad_idx)
    causal = create_causal_mask(tokens.size(1), tokens.device)
    return padding & causal
