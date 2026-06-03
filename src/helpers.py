def count_percent_of_dict(counter: dict):
    """
    Count the % of the total corpus covered by the words.

    eng: 11469 words cover 95% of tokens, 25803 words cover 98% of tokens
    esp: 25805 words cover 95% of tokens, 53828 words cover 98% of tokens
    """
    total = sum(counter.values())
    running = 0
    hit_95 = False
    for i, (word, count) in enumerate(counter.most_common()):
        running += count
        if not hit_95 and running / total >= 0.95:
            print(f"{i} words cover 95% of tokens")
            hit_95 = True
        elif hit_95 and running / total >= 0.98:
            print(f"{i} words cover 98% of tokens")
            break
    print(f"{len(counter)} words cover 100% of tokens")