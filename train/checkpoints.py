import glob
import os
import torch

from src.config import (
    CHECKPOINTS_FOLDER,
    SPM_ENG_PREFIX,
    SPM_ESP_PREFIX,
    spm_model_file,
)
from src.tokenizer import Tokenizer


def save_checkpoint(model, timestamp, dir=CHECKPOINTS_FOLDER):
    """Save model weights.

    Vocab is not saved here - it lives in the sentencepiece .model files
    under data/, which are the single source of truth for tokenisation.
    """
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, f"{timestamp}_model.pt")
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model_cls, device, dir=CHECKPOINTS_FOLDER):
    """Load the latest model weights and rebuild vocabs from sentencepiece models."""
    model_files = sorted(glob.glob(os.path.join(dir, "*_model.pt")))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {dir}")
    latest_model = model_files[-1]
    print(f"Loading {latest_model}")

    tokenizer = Tokenizer()
    eng_vocab = tokenizer.load_vocab(spm_model_file(SPM_ENG_PREFIX))
    esp_vocab = tokenizer.load_vocab(spm_model_file(SPM_ESP_PREFIX))

    model = model_cls(
        src_vocab_size=len(eng_vocab),
        tgt_vocab_size=len(esp_vocab),
    ).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    return model, eng_vocab, esp_vocab, tokenizer
