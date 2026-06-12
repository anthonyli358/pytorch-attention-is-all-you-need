import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self,  src_tokens, tgt_tokens):
        self.pairs = list(zip(src_tokens, tgt_tokens))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    @staticmethod
    def pad_batch(batch) -> tuple[torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)

        src_max_len = max(len(s) for s in src_batch)
        tgt_max_len = max(len(t) for t in tgt_batch)

        src_padded = [s + [0] * (src_max_len - len(s)) for s in src_batch]
        tgt_padded = [t + [0] * (tgt_max_len - len(t)) for t in tgt_batch]

        return torch.tensor(src_padded), torch.tensor(tgt_padded)
