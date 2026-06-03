# pytorch-attention-is-all-you-need
Pytorch implementation of [[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Using English -> Spanish sentence pairs from [Tatoeba](https://tatoeba.org/en/downloads).

Plan:
1. Tokeniser + vocab - transform TSV into integer sequences

First we tokenize e.g. "dog" -> 123 (arbitrary number). Purely a lookup table where a larger corpus would have a larger tokenization dictionary.
We can choose the cutoff for word_counts size by selecting the number which gets 95-98% of the corpus.
Words that appear just once or twice never get enough gradient updates for the model to learn a useful embedding so we trim to reduce noise.

Embed this later

2. Embedding + positional encoding - integers to vectors
3. Attention
4. Encoder block - attention + FFN + residuals
5. Decoder block - masked attention + cross-attention + FFN
6. Full model - wire it all together
7. Training loop - loss, optimiser, learning rate schedule
