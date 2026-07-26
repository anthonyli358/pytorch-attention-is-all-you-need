# tests/test_shapes.py
import torch
from src.masks import create_causal_mask, create_padding_mask
from src.multi_head_attention import MultiHeadAttention
from src.transformer import Transformer

BATCH, SEQ, D_MODEL, VOCAB = 2, 7, 512, 1000


def test_attention_output_shape():
    mha = MultiHeadAttention(d_model=D_MODEL)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = mha(x, x, x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_causal_mask_blocks_future():
    mask = create_causal_mask(4, torch.device("cpu"))
    assert mask[0, 0, 0, 1:].sum() == 0  # position 0 must not see positions 1..3
    assert mask[0, 0, 3, :].all()  # position 3 sees everything


def test_transformer_output_shape():
    model = Transformer(src_vocab_size=VOCAB, tgt_vocab_size=VOCAB)
    src = torch.randint(1, VOCAB, (BATCH, SEQ))
    tgt = torch.randint(1, VOCAB, (BATCH, SEQ))
    out = model(src, tgt)
    assert out.shape == (BATCH, SEQ, VOCAB)


def test_padding_is_masked():
    tokens = torch.tensor([[1, 5, 2, 0, 0]])  # two pads
    mask = create_padding_mask(tokens, 0)
    assert mask.shape == (1, 1, 1, 5)
    assert not mask[0, 0, 0, 3]  # pad position masked out
