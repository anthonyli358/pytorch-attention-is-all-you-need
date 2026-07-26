"""Single source of truth for paths, special tokens, and hyperparameters."""

# ----- Paths -----
DATA_PATH = "data/Sentence pairs in English-Spanish - 2026-06-03.tsv"
CHECKPOINTS_FOLDER = "checkpoints"
SPM_ENG_PREFIX = "data/spm_eng"
SPM_ESP_PREFIX = "data/spm_esp"
TRAIN_STATE_PATH = "data/train_state.pt"

# ----- Special tokens -----
# Indices are fixed and must match the sentencepiece training config
PAD_WORD = "<blank>"  # pad to equal batch length
BOS_WORD = "<bos>"  # beginning of sentence
EOS_WORD = "<eos>"  # end of sentence
UNK_WORD = "<unk>"  # unknown tokens

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ----- Data -----
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1  # remaining 0.1 is test
MAX_LEN_FILTER = 25  # drop sentence pairs longer than this (words)
VOCAB_SIZE = 16000  # BPE subword vocab per language

# ----- Training -----
N_EPOCHS = 25
BATCH_SIZE = 32
LABEL_SMOOTHING = 0.1
WARMUP_RATIO = 0.1  # warmup steps as a fraction of total steps
MIN_WARMUP_STEPS = 200
GRAD_CLIP_NORM = 1.0
PATIENCE = 8  # safety net; best checkpoint saved on each improvement

# ----- Evaluation -----
N_TEST = 500  # held-out sentences to score for BLEU
BEAM_WIDTH = 5
MAX_DECODE_LEN = 50


def spm_model_file(prefix: str, vocab_size: int = VOCAB_SIZE) -> str:
    """Path of the trained sentencepiece model for a given prefix.

    Keeps the filename convention in one place so training and loading
    can never drift apart.
    """
    return f"{prefix}_{vocab_size}.model"
