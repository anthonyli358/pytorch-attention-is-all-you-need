# pytorch-attention-is-all-you-need
Pytorch implementation of [[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Using English -> Spanish sentence pairs from [Tatoeba](https://tatoeba.org/en/downloads).

Plan:
1. Tokeniser + vocab - transform TSV into integer sequences

First we tokenize e.g. "dog" -> 123 (arbitrary number). Purely a lookup table where a larger corpus would have a larger tokenization dictionary.
We can choose the cutoff for word_counts size by selecting the number which gets 95-98% of the corpus.
Words that appear just once or twice never get enough gradient updates for the model to learn a useful embedding so we trim to reduce noise.

We use PAD_WORD later because when we process batches each tensor has uniform dimensions, so with this we can mask them.

2. Embedding + positional encoding - integers to vectors

Initalise an embedding matrix of dimensions len(vocab) x a fixed vector length used for all tokens.
This embeds a sentence.
Now we want to add positional encoding so 'cat' at different positions are encoded differently, each sin and cos at a different frequency (based on position).
We do this by creating a tensor of the same dimensions as the embedding, but encoding position.
This takes a tensor of shapebatch x seq_len

3. Attention

a. Compute raw scores: QK^T / √d_k
b. Mask - before softmax, so masked positions get -1e9 and softmax turns them into ~0 weight
c. Softmax - normalise to get attention weights
d. Dropout - after softmax, randomly zeroes out some attention weights during training
e. Multiply by V
f. Concat and apply W_o

4. Encoder block - attention + FFN + residuals

a. FFN - 3.3 in the paper. Two linear layers with ReLU in between: d_model → d_ff (2048) → d_model.
b. Encoder layer - section 3.1 / Figure 1. N=6 of blocks that does: multi-head self-attention → add residual + layer norm → feed forward → add residual + layer norm.

5. Decoder block - masked multi-head attention + cross-attention + FFN

a. Positional mask stops positions from attending to subsequent positions, this is the 'mask'
b. Cross-attention takes the dot product with the encoder output to find the best match (K), and information about it (V)

6. Full model - wire it all together
7. Training loop - loss, optimiser, learning rate schedule
