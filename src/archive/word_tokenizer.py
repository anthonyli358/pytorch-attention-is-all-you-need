import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from src.config import PAD_WORD, BOS_WORD, EOS_WORD, UNK_WORD


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

    @staticmethod
    def count_percent_of_dict(counter: Counter) -> None:
        """Report how many words cover 95% / 98% of the corpus.

        Used when choosing a word-level vocab cutoff. Kept for reference now
        that tokenisation is subword-based.

        eng: 11469 words cover 95% of tokens, 25803 words cover 98% of tokens
        esp: 25805 words cover 95% of tokens, 53828 words cover 98% of tokens
        """
        total = sum(counter.values())
        running = 0
        hit_95 = False
        for i, (_word, count) in enumerate(counter.most_common()):
            running += count
            if not hit_95 and running / total >= 0.95:
                print(f"{i} words cover 95% of tokens")
                hit_95 = True
            elif hit_95 and running / total >= 0.98:
                print(f"{i} words cover 98% of tokens")
                break
        print(f"{len(counter)} words cover 100% of tokens")