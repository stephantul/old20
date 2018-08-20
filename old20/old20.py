"""Calculate the OLD score for a given n."""
import numpy as np

from pyxdameraulevenshtein import damerau_levenshtein_distance
from functools import partial
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict


def num_combinations(n, k):
    """Calculate the number of combinations."""
    return (n * (n - 1)) / k


def old_all(words, show_progressbar=True):
    """Calculate the OLD distance over all words."""
    return np.mean(old_subloop(words, show_progressbar), 1)


def old_subloop(words,
                show_progressbar,
                function=damerau_levenshtein_distance,
                n=20):
    """Calculate the distance from each word to each other word."""
    old_words = np.zeros((len(words), n)) + np.inf
    maxes = np.max(old_words, 1)
    max_idx = defaultdict(int)
    # Levenshtein distance is symmetric
    # So we only need to calculate the distance
    # between each pair once.
    total = num_combinations(len(words), 2)
    for a, b in tqdm(combinations(np.arange(len(words)), 2),
                     total=total,
                     disable=not show_progressbar):
        dist = function(words[a], words[b])
        if maxes[a] > dist:
            row = old_words[a]
            row[max_idx[a]] = dist
            m = np.argmax(row)
            max_idx[a] = m
            maxes[a] = row[m]
        if maxes[b] > dist:
            row = old_words[b]
            row[max_idx[b]] = dist
            m = np.argmax(row)
            max_idx[b] = m
            maxes[b] = row[m]

    return old_words


def old_n(words,
          n,
          show_progressbar=True,
          function=damerau_levenshtein_distance):
    """
    Calculate the OLD distance for a given n.

    Parameters
    ----------
    words : list
        A list of strings, representing all the types in your corpus.
    n : int

    Returns
    -------
    The old score for a given n.

    """
    if n <= 0:
        raise ValueError("n should be positive.")
    if len(words) <= n:
        raise ValueError("The number of words you have is lower than or equal "
                         "to n. Please lower n.")
    if len(set(words)) != len(words):
        raise ValueError("There are duplicates in your dataset. Please remove"
                         " these, as they will make your estimates unreliable")
    vals = old_subloop(words,
                       show_progressbar,
                       function,
                       n)

    return dict(zip(words, np.sort(vals, axis=1).mean(1)))


old20 = partial(old_n, n=20)
