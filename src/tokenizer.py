import os
import sentencepiece as spm

from src.config import (
    PAD_WORD,
    BOS_WORD,
    EOS_WORD,
    UNK_WORD,
    PAD_IDX,
    BOS_IDX,
    EOS_IDX,
    UNK_IDX,
    VOCAB_SIZE,
    spm_model_file,
)


class SPVocab:
    """Wraps a trained sentencepiece model but behaves like the old vocab dict
    where the codebase expects it: len(vocab) and vocab[SPECIAL_WORD].
    """

    def __init__(self, sp):
        self.sp = sp
        self._special = {
            PAD_WORD: PAD_IDX,
            BOS_WORD: BOS_IDX,
            EOS_WORD: EOS_IDX,
            UNK_WORD: UNK_IDX,
        }

    def __len__(self):
        return self.sp.get_piece_size()

    def __getitem__(self, key):
        if key in self._special:
            return self._special[key]
        return self.sp.piece_to_id(key)


class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def create_vocab(sentences, max_size=VOCAB_SIZE, model_prefix="data/spm"):
        """Train (or load if cached) a sentencepiece BPE model.

        Returns an SPVocab supporting len() and [SPECIAL_WORD] indexing.
        max_size maps to sentencepiece's vocab_size.
        """
        model_file = spm_model_file(model_prefix, max_size)

        if not os.path.exists(model_file):
            corpus = f"{model_prefix}_{max_size}_corpus.txt"
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            with open(corpus, "w", encoding="utf-8") as f:
                for s in sentences:
                    f.write(s.strip() + "\n")

            spm.SentencePieceTrainer.train(
                input=corpus,
                model_prefix=f"{model_prefix}_{max_size}",
                vocab_size=max_size,
                pad_id=PAD_IDX,
                bos_id=BOS_IDX,
                eos_id=EOS_IDX,
                unk_id=UNK_IDX,
                pad_piece=PAD_WORD,
                bos_piece=BOS_WORD,
                eos_piece=EOS_WORD,
                unk_piece=UNK_WORD,
                character_coverage=1.0,
                model_type="bpe",
            )
            os.remove(corpus)

        return Tokenizer.load_vocab(model_file)

    @staticmethod
    def load_vocab(model_file):
        """Load an already-trained sentencepiece model into an SPVocab."""
        sp = spm.SentencePieceProcessor(model_file=model_file)
        return SPVocab(sp)

    @staticmethod
    def tokenize(sentence, vocab):
        """Encode a sentence into subword ids, wrapped with BOS/EOS."""
        ids = vocab.sp.encode(sentence, out_type=int)
        return [vocab[BOS_WORD]] + ids + [vocab[EOS_WORD]]

    @staticmethod
    def detokenize(ids, vocab):
        """Turn token ids back into a string, stripping special tokens.

        Used in inference instead of an idx->word dict, since sentencepiece
        correctly stitches subwords (e.g. 'sent' + 'ó' -> 'sentó').
        """
        specials = {vocab[PAD_WORD], vocab[BOS_WORD], vocab[EOS_WORD]}
        ids = [i for i in ids if i not in specials]
        return vocab.sp.decode(ids)
