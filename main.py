from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.helpers import load_data
from src.config import DATA_PATH, TRAIN_SPLIT, VAL_SPLIT
from src.tokenizer import Tokenizer
from src.transformer import Transformer
from train.translation_dataset import TranslationDataset
from train.warmup_scheduler import WarmupScheduler
from train.train import train, evaluate
from train.inference import greedy_decode, beam_search_decode
from train.metrics import compute_bleu
from train.save_checkpoints import load_checkpoint, save_checkpoint, save_training_state, load_training_state
from train.plot import plot_losses

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = "data/model.pt"
ENG_VOCAB_PATH = "data/eng_vocab.json"
ESP_VOCAB_PATH = "data/esp_vocab.json"
TRAIN_MODEL = True
N_EPOCHS = 20
CKPT = "data/train_state.pt" 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not TRAIN_MODEL:
        model, eng_vocab, esp_vocab, tokenizer = load_checkpoint(Transformer, device)
    else:
        # Data
        data = load_data(DATA_PATH)
        # data = data.head(20000)  # Reduce size for testing
        print("Preprocessing data...")

        # Tokenize
        tokenizer = Tokenizer()
        eng_vocab = tokenizer.create_vocab(data["eng"].tolist(), max_size=15000)
        esp_vocab = tokenizer.create_vocab(data["esp"].tolist(), max_size=30000)
        eng_tokens = [tokenizer.tokenize(s, eng_vocab) for s in data["eng"]]
        esp_tokens = [tokenizer.tokenize(t, esp_vocab) for t in data["esp"]]

        # Train/val split
        train_idx = int(len(data) * TRAIN_SPLIT)
        val_idx = int(len(data) * (TRAIN_SPLIT + VAL_SPLIT))
        train_dataset = TranslationDataset(eng_tokens[:train_idx], esp_tokens[:train_idx])
        val_dataset = TranslationDataset(eng_tokens[train_idx:val_idx], esp_tokens[train_idx:val_idx])
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

        # Model
        model = Transformer(
            src_vocab_size=len(eng_vocab),
            tgt_vocab_size=len(esp_vocab),
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Loss, optimizer, scheduler
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD_WORD
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )  # params from paper
        scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=4000)

        # Training
        start = time.time()
        patience = 3
        start_epoch, best_val_loss, no_improve, train_losses, val_losses = \
            load_training_state(CKPT, model, optimizer, scheduler, device)
        for epoch in range(start_epoch, N_EPOCHS + 1):
            train_loss = train(
                model, train_loader, optimizer, scheduler, loss_fn, device, epoch
            )
            val_loss = evaluate(model, val_loader, loss_fn, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch} | Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} in {(time.time() - start)/60:.1f} mins"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping, bug or reached min loss...")
                    break
        
            save_training_state(CKPT, model, optimizer, scheduler, epoch,
                        best_val_loss, no_improve, train_losses, val_losses)  # resume point
        
        # Save after training
        plot_losses(train_losses, val_losses, f"{TIMESTAMP}_loss.png")
        save_checkpoint(model, eng_vocab, esp_vocab, TIMESTAMP)

    # Test a group of translations
    test_sentences = [
        ("The cat sat on the mat", "El gato se sentó en la alfombra"),
        ("I love you", "Te amo"),
        ("Where is the bathroom", "Dónde está el baño"),
        ("She went to the store", "Ella fue a la tienda"),
        ("We are learning Spanish", "Estamos aprendiendo español"),
    ]

    print("\n--- Translations ---")
    rows = []
    for eng, esp in test_sentences:
        tokens = torch.tensor(tokenizer.tokenize(eng, eng_vocab)).unsqueeze(0)
        greedy = greedy_decode(model, tokens, esp_vocab, device=device)
        beam = beam_search_decode(model, tokens, esp_vocab, device=device)
        rows.append({"source": eng, "target": esp, "greedy": greedy, "beam": beam})
    df = pd.DataFrame(rows)
    print(df)

    greedy_bleu = compute_bleu(df['expected'], df['greedy'])
    beam_bleu = compute_bleu(df['expected'], df['beam'])
    print(f"\nGreedy BLEU: {greedy_bleu}, Beam BLEU: {beam_bleu}")
