from sacrebleu.metrics import BLEU


def compute_bleu(references, translations):
    """
    BLEAU computes n-gram overlap with n=1 to 4 with a length penality
    if shorter than the target.

    Below 10 is useless and above 40 is considered high quality.

    Args:
        references: list of reference strings ["El gato se sentó", ...]
        translations: list of model output strings ["el gato sentó", ...]
    Returns:
        BLEU score
    """
    bleu = BLEU()
    results = bleu.corpus_score(translations, [references])
    return results
