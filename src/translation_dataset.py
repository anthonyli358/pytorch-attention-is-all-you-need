from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self,  src_tokens, tgt_tokens, src_vocab, tgt_vocab):
        self.pairs = list(zip(src_tokens, tgt_tokens))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
