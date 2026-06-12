import torch

from src.config import BOS_WORD, EOS_WORD, UNK_WORD


def greedy_decode(model, src_tokens, tgt_vocab, max_len=50, device="cpu"):
    """Generate a translation token by token."""
    model.eval()
    # Reverse vocab: index -> word
    idx_to_word = {v: k for k, v in tgt_vocab.items()}

    src = src_tokens.to(device)
    tgt = torch.tensor([[tgt_vocab[BOS_WORD]]]).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)  # (1, current_len, vocab_size)
            next_token = output[:, -1].argmax(
                dim=-1, keepdim=True
            )  # greedy pick (highest value), alternative is beam search
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == tgt_vocab[EOS_WORD]:
                break

    tokens = tgt.squeeze(0).tolist()[1:]  # Flatten, skip BOS_WORD
    words = [idx_to_word.get(t, UNK_WORD) for t in tokens if t != tgt_vocab[EOS_WORD]]
    return " ".join(words)
