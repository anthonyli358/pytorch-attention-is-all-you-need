import pandas as pd

from src.config import DATA_PATH, MAX_LEN_FILTER, TRAIN_SPLIT, VAL_SPLIT


def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Read the Tatoeba TSV export into a dataframe with eng/esp columns."""
    return pd.read_csv(
        file_path,
        sep="\t",
        on_bad_lines="skip",
        header=None,
        names=["eng_id", "eng", "esp_id", "esp"],
    )


def filter_by_length(data: pd.DataFrame, max_len: int = MAX_LEN_FILTER) -> pd.DataFrame:
    """Drop pairs where either side exceeds max_len words.

    Long sentences dominate padding and are harder for a small model, so
    removing them speeds up training and improves quality.
    """
    mask = (data["eng"].str.split().str.len() <= max_len) & (
        data["esp"].str.split().str.len() <= max_len
    )
    return data[mask].reset_index(drop=True)


def split_indices(
    n: int, train_split: float = TRAIN_SPLIT, val_split: float = VAL_SPLIT
) -> tuple:
    """
    Deterministic so that training and evaluation runs see the same held-out
    test set without needing to persist it.

    Returns:
        tuple: Return (train_idx, val_idx) boundaries for a deterministic 3-way split.
    """
    train_idx = int(n * train_split)
    val_idx = int(n * (train_split + val_split))
    return train_idx, val_idx


def prepare_data(file_path: str = DATA_PATH):
    """Load, filter, and compute split boundaries in one call.

    Returns:
        (data, train_idx, val_idx)
    """
    data = filter_by_length(load_data(file_path))
    train_idx, val_idx = split_indices(len(data))
    return data, train_idx, val_idx
