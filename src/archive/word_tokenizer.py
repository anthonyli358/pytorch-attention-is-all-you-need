import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from src.config import PAD_WORD, BOS_WORD, EOS_WORD, UNK_WORD
from src.helpers import count_percent_of_dict


class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def create_vocab(sentences:list, max_size:int=15000) -> dict:
        """
        Tokenize a list of sentences into vocab dictionary.
        Keep top max_size words by frequency. Dropped/unseen words fallback to UNK_WORD at encoding.
        """
        word_counts = Counter()
        for s in sentences:
            word_counts.update(s.lower().split())

        most_common = word_counts.most_common(max_size)
        vocab = {PAD_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}
        for word, _ in most_common:
            vocab[word] = len(vocab)
            
        return vocab

    @staticmethod
    def tokenize(sentence:str, vocab:dict) -> list:
        """
        Encode a sentence into tokens. 
        """
        tokens = [vocab[BOS_WORD]]
        tokens += [vocab.get(w, vocab[UNK_WORD]) for w in sentence.lower().split()]
        tokens += [vocab[EOS_WORD]]
        return tokens
    
