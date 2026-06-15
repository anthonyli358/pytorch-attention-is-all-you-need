# pytorch-attention-is-all-you-need
Pytorch implementation of [[1706.03762] Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Using English -> Spanish sentence pairs from [Tatoeba](https://tatoeba.org/en/downloads).

For GPU, got to the [pytorch](https://pytorch.org/get-started/locally/) website and select the local installs to get the bash command.

To use this repo, [install uv](https://docs.astral.sh/uv/getting-started/installation/), pip is recommended.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```


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

a. Pass the batched tokens in the input, not the workaround. Otherwise masks don't apply properly

7. Training loop - loss, optimiser, learning rate schedule

 Approx 25,000 source and target tokens per batch, start with 32.
 
 Extensions

# Create separate file for the test set evaluation

- BLEU score — loss tells you how wrong the model is, but BLEU measures translation quality by comparing n-gram overlap with reference translations. It's the standard metric for machine translation. nltk or sacrebleu libraries have it built in.
- Translate multiple test sentences — run greedy_decode on a batch of test examples and print them side by side with the reference. Eyeballing actual translations tells you more than any number.
- Beam search — replace greedy decode with beam search for better translations.