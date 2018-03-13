"""Calculate the OLD score for a given n."""
from pyxdameraulevenshtein import damerau_levenshtein_distance
from functools import partial
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict
from math import factorial


def num_combinations(n, k):
    """Calculate the number of combinations."""
    return factorial(n) / (factorial(k) * factorial(n - k))


def old_all(words, show_progressbar=True):
    """Calculate the OLD distance over all words."""
    return old_n(words, len(words), show_progressbar=show_progressbar)


def old_n(words, n, show_progressbar=True):
    """
    Calculate the OLD distance for a given n.

    Parameters
    ----------
    words : list
        A list of strings, representing all the types in your corpus.
    n : int
        The number of neighbors to check

    Returns
    -------
    The old score for a given n.

    """
    if n <= 0:
        raise ValueError("n should be a positive number.")
    if len(words) <= n:
        raise ValueError("The number of words you have is lower than or equal "
                         "to the value of your n. Please lower n.")
    if len(set(words)) != len(words):
        raise ValueError("There are duplicates in your dataset. Please remove"
                         " these, as they will make your estimates unreliable")

    old_words = defaultdict(list)

    # Damerau-Levenshtein distance is symmetric
    # So we only need to calculate the distance
    # between each pair once.
    total = num_combinations(len(words), 2)
    for a, b in tqdm(combinations(words, 2),
                     total=total,
                     disable=not show_progressbar):
        dist = damerau_levenshtein_distance(a, b)
        old_words[a].append(dist)
        old_words[b].append(dist)

    return {k: sum(sorted(v)[:n]) / n for k, v in old_words.items()}


old20 = partial(old_n, n=20)
