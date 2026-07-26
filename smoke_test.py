import torch

from src.config import DATA_PATH, UNK_WORD
from src.helpers import load_data, create_padding_mask
from src.tokenizer import Tokenizer
from src.embedder import Embedder
from src.multi_head_attention import MultiHeadAttention
from src.encoder import Encoder
from src.decoder import Decoder
from src.transformer import Transformer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load
    tokenizer = Tokenizer()
    data = load_data(DATA_PATH)
    eng_vocab = tokenizer.create_vocab(data["eng"].tolist(), max_size=15000)
    esp_vocab = tokenizer.create_vocab(data["esp"].tolist(), max_size=30000)

    # Tokenize
    test_tokens = tokenizer.tokenize("Cat sat on a rug", eng_vocab)
    test_tgt_tokens = tokenizer.tokenize("El gato se sentó en la alfombra", esp_vocab)
    eng_vocab_size = len(eng_vocab)
    esp_vocab_size = len(esp_vocab)
    print(test_tokens)

    # Batch the 1d tokens
    test_tokens = torch.tensor(test_tokens).unsqueeze(0)
    test_tgt_tokens = torch.tensor(test_tgt_tokens).unsqueeze(0)

    # Embed
    eng_embedder = Embedder(eng_vocab_size)
    eng_embedding = eng_embedder(test_tokens)
    esp_embedder = Embedder(esp_vocab_size)
    esp_embedding = esp_embedder(test_tgt_tokens)
    print(f"Embedding: {eng_embedding.shape}")

    # Test MHA
    mha_layer = MultiHeadAttention()
    mask = create_padding_mask(test_tokens, eng_vocab[UNK_WORD])
    attention = mha_layer(q=eng_embedding, k=eng_embedding, v=eng_embedding, mask=None)
    print(f"Attention: {attention.shape}")

    # Encode & Decode
    encoder = Encoder(vocab_size=eng_vocab_size)
    encoder_output = encoder(test_tokens)
    print(f"Encoder: {encoder_output.shape}")
    decoder = Decoder(vocab_size=esp_vocab_size)
    decoder_output = decoder(test_tgt_tokens, encoder_output)
    print(f"Decoder: {decoder_output.shape}")

    # Transformer
    transformer = Transformer(eng_vocab_size, esp_vocab_size)
    transformer_output = transformer(test_tokens, test_tgt_tokens)
    print(f"Transformer: {transformer_output.shape}")
