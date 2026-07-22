import os
import sentencepiece as spm
from src.config import PAD_WORD, BOS_WORD, EOS_WORD, UNK_WORD


class SPVocab:
    """
    Wraps a trained sentencepiece model but behaves like the old vocab dict
    where the codebase expects it: len(vocab) and vocab[SPECIAL_WORD].
    """
    def __init__(self, sp):
        self.sp = sp
        # Special tokens are trained at fixed ids to match the old dict layout
        self._special = {PAD_WORD: 0, BOS_WORD: 1, EOS_WORD: 2, UNK_WORD: 3}

    def __len__(self):
        return self.sp.get_piece_size()

    def __getitem__(self, key):
        # Supports vocab[BOS_WORD] etc, same as the old dict
        if key in self._special:
            return self._special[key]
        return self.sp.piece_to_id(key)


class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def create_vocab(sentences, max_size=15000, model_prefix="data/spm"):
        """
        Train (or load) a sentencepiece BPE model. Returns an SPVocab that
        supports len() and [SPECIAL_WORD] indexing like the old vocab dict.
        max_size maps to sentencepiece's vocab_size.
        """
        model_file = f"{model_prefix}_{max_size}.model"

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
                pad_id=0, bos_id=1, eos_id=2, unk_id=3,
                pad_piece=PAD_WORD, bos_piece=BOS_WORD,
                eos_piece=EOS_WORD, unk_piece=UNK_WORD,
                character_coverage=1.0,
                model_type="bpe",
            )
            os.remove(corpus)

        sp = spm.SentencePieceProcessor(model_file=model_file)
        return SPVocab(sp)

    @staticmethod
    def tokenize(sentence, vocab):
        """
        Encode a sentence into subword token ids, wrapped with BOS/EOS.
        Same signature and return type (list of ints) as before.
        """
        ids = vocab.sp.encode(sentence, out_type=int)
        return [vocab[BOS_WORD]] + ids + [vocab[EOS_WORD]]

    @staticmethod
    def detokenize(ids, vocab):
        """
        Turn token ids back into a string, stripping special tokens.
        Use this in inference instead of the idx_to_word dict lookup.
        """
        specials = {vocab[PAD_WORD], vocab[BOS_WORD], vocab[EOS_WORD]}
        ids = [i for i in ids if i not in specials]
        return vocab.sp.decode(ids)

    @staticmethod
    def load_vocab(model_file):
        """Load an already-trained sentencepiece model into an SPVocab."""
        sp = spm.SentencePieceProcessor(model_file=model_file)
        return SPVocab(sp)