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

Removed unk from greedy_decode and beam_search_decode due to underfitting (missing rare words like alfombra and bano). We can run more epochs (> 20) or try proper subword tokenization to better capture rarer words.

Add label smoothing, and reduce dropout for poc (smaller dataset)

# pytorch-attention-is-all-you-need

A from-scratch PyTorch implementation of the Transformer from [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), trained for English → Spanish translation on [Tatoeba](https://tatoeba.org/en/downloads) sentence pairs.

Every core component — scaled dot-product attention, multi-head attention, positional encoding, the encoder/decoder stack, masking, and beam search — is written by hand rather than using `torch.nn.Transformer`. The only external model dependency is [SentencePiece](https://github.com/google/sentencepiece) for subword tokenisation.

## Results

Trained for 25 epochs on ~single GPU, the model reaches **22.5 BLEU** on a 500-sentence held-out test set.

| Source | Reference | Model output |
|---|---|---|
| The cat sat on the mat | El gato se sentó en la alfombra | El gato se sentó en la silla. |
| I love you | Te amo | Te quiero. |
| Where is the bathroom | Dónde está el baño | ¿Dónde está el baño? |
| She went to the store | Ella fue a la tienda | Ella fue a la tienda. |
| We are learning Spanish | Estamos aprendiendo español | Estamos aprendiendo español. |

The model produces fluent, correctly punctuated Spanish, including inverted question marks it learned from the data. Some outputs differ from the reference while remaining valid ("Te quiero" vs "Te amo" for "I love you") — a known limitation of BLEU, which penalises correct alternatives.

## Architecture

Faithful to the base model in the paper:

- **d_model**: 512
- **Layers**: 6 encoder, 6 decoder
- **Attention heads**: 8 (d_k = 64 per head)
- **Feed-forward dimension**: 2048
- **Dropout**: 0.1
- **Positional encoding**: fixed sinusoidal

Training details follow the paper where practical: Adam with β = (0.9, 0.98), the Noam warmup learning-rate schedule, label smoothing (0.1), and Xavier initialisation. The warmup step count is scaled to the size of the training run rather than fixed at the paper's 4000, which assumes a much larger dataset.

## Project structure

```
src/
  embedder.py               token embedding + sinusoidal positional encoding
  multi_head_attention.py   scaled dot-product + multi-head attention
  positionwise_feedforward.py
  encoder.py                encoder layer + stack
  decoder.py                decoder layer (self-attn, cross-attn, ffn) + stack
  transformer.py            full model, wires encoder + decoder + output projection
  tokenizer.py              SentencePiece BPE wrapper
  helpers.py                data loading, padding + causal masks
  config.py                 hyperparameters and special tokens
train/
  translation_dataset.py    Dataset + dynamic batch padding
  train.py                  train / evaluate loops
  warmup_scheduler.py       Noam learning-rate schedule
  inference.py              greedy + beam search decoding
  metrics.py                BLEU (sacrebleu)
  save_checkpoints.py       save / load model weights
  plot.py                   loss curves
main.py                     entry point (train or load + evaluate)
```

## Usage

Install dependencies:

```bash
uv sync
```

Download the English–Spanish sentence pairs from [Tatoeba](https://tatoeba.org/en/downloads) and place the TSV in `data/`, updating `DATA_PATH` in `src/config.py`.

Train from scratch (set `TRAIN_MODEL = True` in `main.py`):

```bash
python main.py
```

The best checkpoint (by validation loss) is saved to `checkpoints/` each time it improves, and a loss curve is written alongside it. To evaluate an existing checkpoint, set `TRAIN_MODEL = False` and run again.

## Implementation notes

A few details that matter and are easy to get wrong:

- **Masking**: a padding mask (shape `(batch, 1, 1, seq_len)`) prevents attention over `<pad>` tokens in the encoder and cross-attention; the decoder additionally combines it with a causal mask so each position only attends to earlier ones. Masked positions are set to `-1e9` *before* softmax.
- **Teacher forcing**: the decoder input is the target shifted right (`<bos> ...`) and the loss compares against the target shifted left (`... <eos>`), so each position predicts the next token.
- **Subword tokenisation**: word-level tokenisation sent rare words to `<unk>` and capped translation quality. SentencePiece BPE decomposes unseen words into known subword pieces, effectively eliminating `<unk>`.
- **Checkpoint saving**: the best model is saved *during* training on each validation improvement, so a crash or interruption never loses good weights.

## Acknowledgements

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- [Tatoeba](https://tatoeba.org) for the sentence-pair corpus
- [SentencePiece](https://github.com/google/sentencepiece) for subword tokenisation