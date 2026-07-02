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
    references = [str(r) for r in references]
    translations = [str(t) for t in translations]
    print(f"DEBUG types: refs={type(references)}, first={type(references[0]) if references else 'empty'}")
    print(f"DEBUG refs: {references}")
    print(f"DEBUG hyps: {translations}")
    bleu = BLEU()
    return bleu.corpus_score(translations, [references])
