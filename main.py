import torch

from src.config import DATA_PATH
from src.tokenizer import Tokenizer
from src.embedder import Embedder


if __name__ == "__main__":
    tokenizer = Tokenizer()
    data = tokenizer.load_data(DATA_PATH)
    eng_tokens = tokenizer.tokenize(data['eng'].tolist(), max_size=15000)
    esp_tokens = tokenizer.tokenize(data['esp'].tolist(), max_size=30000)
    test_tokens = tokenizer.encode("Cat sat on a rug", eng_tokens)
    print(test_tokens)
    eng_embedder = Embedder(len(eng_tokens))
    eng_embedding = eng_embedder(torch.tensor(test_tokens))  # set to a batch
    print(eng_embedding)
