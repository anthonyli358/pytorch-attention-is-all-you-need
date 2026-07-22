import glob
import os
import torch
from src.config import CHECKPOINTS_FOLDER
from src.tokenizer import Tokenizer

ENG_SPM = "data/spm_eng_16000.model"
ESP_SPM = "data/spm_esp_16000.model"


def save_checkpoint(model, eng_vocab, esp_vocab, timestamp, dir=CHECKPOINTS_FOLDER):
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dir, f"{timestamp}_model.pt"))
    print(f"Checkpoint saved to {dir}/{timestamp}_model.pt")


def load_checkpoint(model_cls, device, dir=CHECKPOINTS_FOLDER):
    model_files = sorted(glob.glob(os.path.join(dir, "*_model.pt")))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {dir}")
    latest_model = model_files[-1]
    print(f"Loading {latest_model}")

    tokenizer = Tokenizer()
    eng_vocab = tokenizer.load_vocab(ENG_SPM)
    esp_vocab = tokenizer.load_vocab(ESP_SPM)

    model = model_cls(
        src_vocab_size=len(eng_vocab),
        tgt_vocab_size=len(esp_vocab),
    ).to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    return model, eng_vocab, esp_vocab, tokenizer
