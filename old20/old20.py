"""Calculate the OLD score for a given n."""
from pyxdameraulevenshtein import damerau_levenshtein_distance
from functools import partial
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict


def old_n(words, n):
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
    words = defaultdict(list)

    # Damerau-Levenshtein distance is symmetric
    # So we only need to calculate the distance
    # between each pair once.
    for a, b in tqdm(combinations(words, 2)):
        dist = damerau_levenshtein_distance(a, b)
        words[a].append(dist)
        words[b].append(dist)

    return {k: list(sorted(v))[:n] / n for k, v in words.items()}


old20 = partial(old_n, n=20)
