from datetime import datetime
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    VOCAB_SIZE,
    N_EPOCHS,
    BATCH_SIZE,
    PATIENCE,
    N_TEST,
)
from src.data import prepare_data
from src.tokenizer import Tokenizer
from src.transformer import Transformer
from train.translation_dataset import TranslationDataset
from train.warmup_scheduler import WarmupScheduler
from train.train import train, evaluate
from evals.inference import greedy_decode, beam_search_decode
from evals.metrics import compute_bleu
from train.checkpoints import load_checkpoint, save_checkpoint
from train.plot import plot_losses

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TRAIN_MODEL = False


def run_training(data, train_idx, val_idx, device):
    """Train from scratch, saving the best checkpoint. Returns model + vocabs + tokenizer."""
    tokenizer = Tokenizer()
    eng_vocab = tokenizer.create_vocab(
        data["eng"].tolist(), max_size=VOCAB_SIZE, model_prefix="data/spm_eng"
    )
    esp_vocab = tokenizer.create_vocab(
        data["esp"].tolist(), max_size=VOCAB_SIZE, model_prefix="data/spm_esp"
    )
    print(
        f"Round trip: {tokenizer.detokenize(tokenizer.tokenize('El gato se sentó', esp_vocab), esp_vocab)!r}"
    )

    eng_tokens = [tokenizer.tokenize(s, eng_vocab) for s in data["eng"]]
    esp_tokens = [tokenizer.tokenize(t, esp_vocab) for t in data["esp"]]

    train_dataset = TranslationDataset(eng_tokens[:train_idx], esp_tokens[:train_idx])
    val_dataset = TranslationDataset(
        eng_tokens[train_idx:val_idx], esp_tokens[train_idx:val_idx]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=TranslationDataset.pad_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=TranslationDataset.pad_batch,
    )

    model = Transformer(
        src_vocab_size=len(eng_vocab), tgt_vocab_size=len(esp_vocab)
    ).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    total_steps = len(train_loader) * N_EPOCHS
    warmup_steps = max(200, int(total_steps * 0.1))
    print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
    scheduler = WarmupScheduler(optimizer, d_model=512, warmup_steps=warmup_steps)

    start = time.time()
    best_val_loss = float("inf")
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train(
            model, train_loader, optimizer, scheduler, loss_fn, device, epoch
        )
        val_loss = evaluate(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch} | Train: {train_loss:.3f} | Val: {val_loss:.3f} | {(time.time() - start)/60:.1f} mins"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(model, TIMESTAMP)
            print(f"  New best val loss {best_val_loss:.3f} - checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping (safety net).")
                break

    plot_losses(train_losses, val_losses, f"{TIMESTAMP}_loss.png")
    return model, eng_vocab, esp_vocab, tokenizer


def show_sample_translations(model, eng_vocab, esp_vocab, tokenizer, device):
    """Run trained model for sample vocab"""
    test_sentences = [
        ("The cat sat on the mat", "El gato se sentó en la alfombra"),
        ("I love you", "Te amo"),
        ("Where is the bathroom", "Dónde está el baño"),
        ("She went to the store", "Ella fue a la tienda"),
        ("We are learning Spanish", "Estamos aprendiendo español"),
    ]
    rows = []
    for eng, esp in test_sentences:
        tokens = torch.tensor(tokenizer.tokenize(eng, eng_vocab)).unsqueeze(0)
        rows.append(
            {
                "source": eng,
                "target": esp,
                "greedy": greedy_decode(model, tokens, esp_vocab, device=device),
                "beam": beam_search_decode(model, tokens, esp_vocab, device=device),
            }
        )
    print("\n--- Sample Translations ---")
    print(pd.DataFrame(rows).to_string(index=False))


def evaluate_test_bleu(
    model, eng_vocab, esp_vocab, tokenizer, test_eng, test_esp, device
):
    """Evaluate BLEU score for hold out set."""
    n = min(N_TEST, len(test_eng))
    refs, hyps = [], []
    for i in range(n):
        tokens = torch.tensor(tokenizer.tokenize(test_eng[i], eng_vocab)).unsqueeze(0)
        hyps.append(greedy_decode(model, tokens, esp_vocab, device=device))
        refs.append(test_esp[i])
    print(f"\n--- Held-out Test BLEU ({n} sentences) ---")
    print(compute_bleu(refs, hyps))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, train_idx, val_idx = prepare_data()
    print(f"Using {len(data)} sentence pairs after length filter")
    test_eng = data["eng"][val_idx:].tolist()
    test_esp = data["esp"][val_idx:].tolist()

    if TRAIN_MODEL:
        model, eng_vocab, esp_vocab, tokenizer = run_training(
            data, train_idx, val_idx, device
        )
    else:
        model, eng_vocab, esp_vocab, tokenizer = load_checkpoint(Transformer, device)

    show_sample_translations(model, eng_vocab, esp_vocab, tokenizer, device)
    evaluate_test_bleu(
        model, eng_vocab, esp_vocab, tokenizer, test_eng, test_esp, device
    )
