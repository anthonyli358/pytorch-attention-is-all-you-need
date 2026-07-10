from datetime import datetime
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
from train.save_checkpoints import load_checkpoint, save_checkpoint
from train.plot import plot_losses

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAIN_MODEL = True
N_EPOCHS = 25
BATCH_SIZE = 32
VOCAB_SIZE = 16000       # subword vocab per language
# MAX_LEN_FILTER = 25      # drop pairs longer than this (words) for cleaner training
PATIENCE = 8             # safety net only; best checkpoint saved each improvement


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not TRAIN_MODEL:
        model, eng_vocab, esp_vocab, tokenizer = load_checkpoint(Transformer, device)
    else:
        # Data
        data = load_data(DATA_PATH)

        # Filter very long sentences - cleaner training, less padding waste
        # mask = (
        #     (data["eng"].str.split().str.len() <= MAX_LEN_FILTER)
        #     & (data["esp"].str.split().str.len() <= MAX_LEN_FILTER)
        # )
        # data = data[mask].reset_index(drop=True)
        print(f"Training on {len(data)} sentence pairs after length filter")

        # Tokenize (sentencepiece BPE, trained once then cached to data/)
        tokenizer = Tokenizer()
        eng_vocab = tokenizer.create_vocab(
            data["eng"].tolist(), max_size=VOCAB_SIZE, model_prefix="data/spm_eng"
        )
        esp_vocab = tokenizer.create_vocab(
            data["esp"].tolist(), max_size=VOCAB_SIZE, model_prefix="data/spm_esp"
        )

        # Round-trip sanity check before a long run
        sample = tokenizer.tokenize("El gato se sentó", esp_vocab)
        print(f"Round trip: {tokenizer.detokenize(sample, esp_vocab)!r}")

        eng_tokens = [tokenizer.tokenize(s, eng_vocab) for s in data["eng"]]
        esp_tokens = [tokenizer.tokenize(t, esp_vocab) for t in data["esp"]]

        # Train/val split
        train_idx = int(len(data) * TRAIN_SPLIT)
        val_idx = int(len(data) * (TRAIN_SPLIT + VAL_SPLIT))
        train_dataset = TranslationDataset(eng_tokens[:train_idx], esp_tokens[:train_idx])
        val_dataset = TranslationDataset(
            eng_tokens[train_idx:val_idx], esp_tokens[train_idx:val_idx]
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=TranslationDataset.pad_batch,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=TranslationDataset.pad_batch,
        )

        # Model
        model = Transformer(
            src_vocab_size=len(eng_vocab),
            tgt_vocab_size=len(esp_vocab),
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Loss (label smoothing per paper 5.4), optimizer, scheduler
        loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        total_steps = len(train_loader) * N_EPOCHS
        warmup_steps = max(200, int(total_steps * 0.1))
        print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
        scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=warmup_steps)

        # Training - save BEST checkpoint each improvement (weights first, always)
        start = time.time()
        best_val_loss = float("inf")
        no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
            val_loss = evaluate(model, val_loader, loss_fn, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch} | Train: {train_loss:.3f} | Val: {val_loss:.3f} "
                f"| {(time.time() - start)/60:.1f} mins"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                save_checkpoint(model, eng_vocab, esp_vocab, TIMESTAMP)  # best so far
                print(f"  New best val loss {best_val_loss:.3f} - checkpoint saved")
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print("Early stopping (safety net).")
                    break

        plot_losses(train_losses, val_losses, f"{TIMESTAMP}_loss.png")

    # ----- Translations (runs in both modes) -----
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
    print(df.to_string(index=False))

    greedy_bleu = compute_bleu(df["target"].tolist(), df["greedy"].tolist())
    beam_bleu = compute_bleu(df["target"].tolist(), df["beam"].tolist())
    print(f"\nGreedy BLEU: {greedy_bleu}")
    print(f"Beam BLEU:   {beam_bleu}")