import glob
import os
import json
import torch
from datetime import datetime

from src.config import CHECKPOINTS_FOLDER


def save_training_state(path, model, optimizer, scheduler, epoch,
                        best_val_loss, no_improve, train_losses, val_losses):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "no_improve": no_improve,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, path)

def load_training_state(path, model, optimizer, scheduler, device):
    """Restore training state in place. Returns (start_epoch, best_val_loss,
    no_improve, train_losses, val_losses). If no checkpoint exists, returns fresh defaults."""
    if not os.path.exists(path):
        return 1, float("inf"), 0, [], []

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resuming from epoch {ckpt['epoch'] + 1}")
    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_loss"],
        ckpt["no_improve"],
        ckpt["train_losses"],
        ckpt["val_losses"],
    )

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
