import glob
import os
import json
import torch
from datetime import datetime

from src.config import CHECKPOINTS_FOLDER


def save_checkpoint(model, eng_vocab, esp_vocab, timestamp, dir=CHECKPOINTS_FOLDER):
    """
    Save with current timestamp
    """
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dir, f"{timestamp}_model.pt"))
    with open(os.path.join(dir, f"{timestamp}_eng_vocab.json"), "w") as f:
        json.dump(eng_vocab, f)
    with open(os.path.join(dir, f"{timestamp}_esp_vocab.json"), "w") as f:
        json.dump(esp_vocab, f)
    print(f"Checkpoints saved to {dir}/")


def load_checkpoint(model_cls, device, dir=CHECKPOINTS_FOLDER):
    """
    Get latest timestamp, and load all models + vocab with that timestamp
    """
    # Find latest model file
    model_files = sorted(glob.glob(os.path.join(dir, "*_model.pt")))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {dir}")
    latest_model = model_files[-1]
    print(f"Loading {latest_model}")

    timestamp = os.path.basename(latest_model).replace("_model.pt", "")
    with open(os.path.join(dir, f"{timestamp}_eng_vocab.json")) as f:
        eng_vocab = json.load(f)
    with open(os.path.join(dir, f"{timestamp}_esp_vocab.json")) as f:
        esp_vocab = json.load(f)

    model = model_cls(
        src_vocab_size=len(eng_vocab),
        tgt_vocab_size=len(esp_vocab),
    ).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    return model, eng_vocab, esp_vocab
