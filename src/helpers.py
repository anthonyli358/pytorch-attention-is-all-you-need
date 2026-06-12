import pandas as pd
import torch

def load_data(file_path:str) -> pd.DataFrame: 
    data = pd.read_csv(
        file_path,
        sep="\t",
        on_bad_lines="skip",
        header=None,
        names=["eng_id", "eng", "esp_id", "esp"],
    )
    return data

def count_percent_of_dict(counter: dict):
    """
    Count the % of the total corpus covered by the words.

    eng: 11469 words cover 95% of tokens, 25803 words cover 98% of tokens
    esp: 25805 words cover 95% of tokens, 53828 words cover 98% of tokens
    """
    total = sum(counter.values())
    running = 0
    hit_95 = False
    for i, (word, count) in enumerate(counter.most_common()):
        running += count
        if not hit_95 and running / total >= 0.95:
            print(f"{i} words cover 95% of tokens")
            hit_95 = True
        elif hit_95 and running / total >= 0.98:
            print(f"{i} words cover 98% of tokens")
            break
    print(f"{len(counter)} words cover 100% of tokens")

def create_padding_mask(tokens:torch.Tensor, mask_idx:int=0):
    mask = (tokens != mask_idx)
    return mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
