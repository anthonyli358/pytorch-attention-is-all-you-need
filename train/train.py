import torch
from torch import nn


def train(model, dataloader, optimizer, scheduler, loss_fn, device, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        # Decoder input: everything except last token  [<bos> El gato se sentó en la alfombra]
        # Target: everything except first token  [El gato se sentó en la alfombra <eos>]
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)  # (batch, seq_len, vocab_size)

        # Reshape for cross entropy: (batch * seq_len, vocab_size) vs (batch * seq_len)
        output = output.reshape(-1, output.size(-1))
        tgt_target = tgt_target.reshape(-1)

        loss = loss_fn(output, tgt_target)
        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient clipping to prevent exploding gradients
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    """_summary_
    Similar to train, but without gradient updating.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            output = model(src, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_target = tgt_target.reshape(-1)

            loss = loss_fn(output, tgt_target)
            total_loss += loss.item()

    return total_loss / len(dataloader)
