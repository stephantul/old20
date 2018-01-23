"""Calculate the OLD score for a given n."""
from pyxdameraulevenshtein import damerau_levenshtein_distance
from functools import partial
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict
from scipy.misc import comb


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
    old_words = defaultdict(list)

    # Damerau-Levenshtein distance is symmetric
    # So we only need to calculate the distance
    # between each pair once.
    total = comb(len(words), 2)
    for a, b in tqdm(combinations(words, 2), total=total):
        dist = damerau_levenshtein_distance(a, b)
        old_words[a].append(dist)
        old_words[b].append(dist)

    return {k: sum(sorted(v)[:n]) / n for k, v in old_words.items()}


old20 = partial(old_n, n=20)
