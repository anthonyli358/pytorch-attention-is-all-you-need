import torch

from src.config import BOS_WORD, EOS_WORD, UNK_WORD


def greedy_decode(model, src_tokens, tgt_vocab, max_len=50, device="cpu"):
    """Generate a translation token by token."""
    model.eval()
    # Reverse vocab: index -> word
    idx_to_word = {v: k for k, v in tgt_vocab.items()}
    src = src_tokens.to(device)
    tgt = torch.tensor([[tgt_vocab[BOS_WORD]]]).to(device)
    unk_idx = tgt_vocab[UNK_WORD]
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)  # (1, current_len, vocab_size)
            logits = output[:, -1]           # (1, vocab_size)
            logits[:, unk_idx] = float('-inf')  # ban <unk>
            next_token = logits.argmax(
                dim=-1, keepdim=True
            )  # greedy pick (highest value), alternative is beam search
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == tgt_vocab[EOS_WORD]:
                break
    tokens = tgt.squeeze(0).tolist()[1:]  # Flatten, skip BOS_WORD
    words = [idx_to_word.get(t, UNK_WORD) for t in tokens if t != tgt_vocab[EOS_WORD]]
    return " ".join(words)


def beam_search_decode(model, src_tokens, tgt_vocab, beam_width=5, max_len=50, device="cpu"):
    """
    Generate a translation using beam search.
    Keeps the top beam_width sentences and explores them all in parallel.
    """
    model.eval()
    idx_to_word = {v: k for k, v in tgt_vocab.items()}
    src = src_tokens.to(device)
    bos_idx = tgt_vocab[BOS_WORD]
    eos_idx = tgt_vocab[EOS_WORD]
    unk_idx = tgt_vocab[UNK_WORD]

    # Each beam: (log_probability, token_sequence)
    beams = [(0.0, [bos_idx])]
    completed = []

    with torch.no_grad():
        for _ in range(max_len):
            candidates = []

            for score, seq in beams:
                tgt = torch.tensor([seq]).to(device)
                output = model(src, tgt)
                logits = output[:, -1]
                logits[0, unk_idx] = float('-inf')   # ban <unk>
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                # Take top beam_width tokens
                top_probs, top_indices = log_probs.topk(beam_width)

                for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                    new_seq = seq + [idx]
                    new_score = score + prob  # sum of negative log probs

                    if idx == eos_idx:
                        # Normalise by length to avoid favouring short sequences
                        completed.append((new_score / len(new_seq), new_seq))
                    else:
                        candidates.append((new_score, new_seq))

            if not candidates:
                break

            # Keep top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    # If no completed sequences, use best beam
    if not completed:
        completed = [(score / len(seq), seq) for score, seq in beams]

    # Pick highest scoring completed sequence
    # List of sequences of (normalised_score, sequence)
    best_seq = sorted(completed, key=lambda x: x[0], reverse=True)[0][1] 

    words = [idx_to_word.get(t, UNK_WORD) for t in best_seq[1:] if t != eos_idx]
    return " ".join(words)