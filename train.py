import torch
from torch import nn
from torch.utils.data import DataLoader

from src.helpers import load_data
from src.config import DATA_PATH, TRAIN_SPLIT
from src.tokenizer import Tokenizer
from src.translation_dataset import TranslationDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    data = load_data(DATA_PATH)

    # Tokenize
    tokenizer = Tokenizer()
    eng_vocab = tokenizer.create_vocab(data["eng"].tolist(), max_size=15000)
    esp_vocab = tokenizer.create_vocab(data["esp"].tolist(), max_size=30000)
    eng_tokens = [tokenizer.encode(s, eng_vocab) for s in data["eng"]]
    esp_tokens = [tokenizer.encode(t, esp_vocab) for t in data["esp"]]

    # Train/val split
    split_idx = int(len(data) * TRAIN_SPLIT)
    train_dataset = TranslationDataset(eng_tokens[:split_idx], esp_tokens[:split_idx])
    val_dataset = TranslationDataset(eng_tokens[split_idx:], esp_tokens[split_idx:])
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=TranslationDataset.pad_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=TranslationDataset.pad_batch,
    )
