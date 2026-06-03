import pandas as pd
from collections import Counter

from config import DATA_PATH, PAD_WORD, BOS_WORD, EOS_WORD, UNK_WORD
from helpers import count_percent_of_dict


class Tokenizer:
    def __init__(self):
        pass

    def load_data(self, file_path:str) -> pd.DataFrame: 
        data = pd.read_csv(
            file_path,
            sep="\t",
            on_bad_lines="skip",
            header=None,
            names=["eng_id", "eng", "esp_id", "esp"],
        )
        return data

    def tokenize(self, sentences:list, max_size:int=15000) -> dict:
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

    def encode(self, sentence:str, vocab:dict) -> list:
        """
        Encode a sentence into tokens. 
        """
        tokens = [vocab[BOS_WORD]]
        tokens += [vocab.get(w, vocab[UNK_WORD]) for w in sentence.lower().split()]
        tokens += [vocab[EOS_WORD]]
        return tokens

if __name__ == "__main__":
    tokenizer = Tokenizer()
    data = tokenizer.load_data(DATA_PATH)
    eng_tokens = tokenizer.tokenize(data['eng'].tolist(), max_size=15000)
    esp_tokens = tokenizer.tokenize(data['esp'].tolist(), max_size=30000)
    print(tokenizer.encode("Cat sat on a rug", eng_tokens))

