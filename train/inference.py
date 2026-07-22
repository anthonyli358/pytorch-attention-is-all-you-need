import torch
from src.config import BOS_WORD, EOS_WORD
from src.tokenizer import Tokenizer


def greedy_decode(model, src_tokens, tgt_vocab, max_len=50, device="cpu"):
    """Generate a translation token by token."""
    model.eval()
    src = src_tokens.to(device)
    tgt = torch.tensor([[tgt_vocab[BOS_WORD]]]).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)  # (1, current_len, vocab_size)
            next_token = output[:, -1].argmax(dim=-1, keepdim=True)  # greedy pick
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == tgt_vocab[EOS_WORD]:
                break

    ids = tgt.squeeze(0).tolist()
    return Tokenizer.detokenize(ids, tgt_vocab)  # handles subword stitching


def beam_search_decode(model, src_tokens, tgt_vocab, beam_width=5, max_len=50, device="cpu"):
    """
    Generate a translation using beam search.
    Keeps the top beam_width sentences and explores them all in parallel.
    """
    model.eval()
    src = src_tokens.to(device)
    bos_idx = tgt_vocab[BOS_WORD]
    eos_idx = tgt_vocab[EOS_WORD]

    beams = [(0.0, [bos_idx])]   # (log_probability, token_sequence)
    completed = []

    with torch.no_grad():
        for _ in range(max_len):
            candidates = []
            for score, seq in beams:
                tgt = torch.tensor([seq]).to(device)
                output = model(src, tgt)
                logits = output[:, -1]  # (1, vocab_size)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                top_probs, top_indices = log_probs.topk(beam_width)
                for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                    new_seq = seq + [idx]
                    new_score = score + prob  # sum of log probs
                    if idx == eos_idx:
                        completed.append((new_score / len(new_seq), new_seq))  # length norm
                    else:
                        candidates.append((new_score, new_seq))

            if not candidates:
                break
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    if not completed:
        completed = [(score / len(seq), seq) for score, seq in beams]

    best_seq = sorted(completed, key=lambda x: x[0], reverse=True)[0][1]
    return Tokenizer.detokenize(best_seq, tgt_vocab)