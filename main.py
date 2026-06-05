import torch

from src.config import DATA_PATH, UNK_WORD
from src.helpers import create_mask
from src.tokenizer import Tokenizer
from src.embedder import Embedder
from src.multi_head_attention import MultiHeadAttention


if __name__ == "__main__":
    tokenizer = Tokenizer()
    data = tokenizer.load_data(DATA_PATH)
    eng_tokens = tokenizer.tokenize(data['eng'].tolist(), max_size=15000)
    esp_tokens = tokenizer.tokenize(data['esp'].tolist(), max_size=30000)
    test_tokens = tokenizer.encode("Cat sat on a rug", eng_tokens)
    print(test_tokens)
    eng_embedder = Embedder(len(eng_tokens))
    eng_embedding = eng_embedder(torch.tensor(test_tokens))
    print(eng_embedding)
    mha_layer = MultiHeadAttention()
    mask = create_mask(eng_embedding, eng_tokens[UNK_WORD])
    attention = mha_layer(q=eng_embedding, k=eng_embedding, v=eng_embedding, mask=None)
    print(attention)
    