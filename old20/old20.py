"""Calculate the OLD score for a given n."""
import numpy as np

from pyxdameraulevenshtein import damerau_levenshtein_distance
from functools import partial
from itertools import combinations
from tqdm import tqdm


def num_combinations(n, k):
    """Calculate the number of combinations."""
    return (n * (n - 1)) / k


def old_all(words, show_progressbar=True):
    """Calculate the OLD distance over all words."""
    return np.mean(old_subloop(words, show_progressbar), 1)


def old_subloop(words,
                show_progressbar,
                function=damerau_levenshtein_distance):
    """Calculate the distance from each word to each other word."""
    old_words = np.zeros((len(words), len(words)))

    # Damerau-Levenshtein distance is symmetric
    # So we only need to calculate the distance
    # between each pair once.
    total = num_combinations(len(words), 2)
    for a, b in tqdm(combinations(np.arange(len(words)), 2),
                     total=total,
                     disable=not show_progressbar):
        dist = function(words[a], words[b])
        old_words[a, b] = dist
        old_words[b, a] = dist

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
    n : list of ints

    Returns
    -------
    The old score for a given n.

    """
    if isinstance(n, int):
        n = [n]
    if any([x <= 0 for x in n]):
        raise ValueError("all values of n should be positive numbers.")
    if any([len(words) <= x for x in n]):
        raise ValueError("The number of words you have is lower than or equal "
                         "to a value of n. Please lower n.")
    if len(set(words)) != len(words):
        raise ValueError("There are duplicates in your dataset. Please remove"
                         " these, as they will make your estimates unreliable")
    vals = old_subloop(words,
                       show_progressbar,
                       function)

    output = []
    for x in n:

        old_vals = np.partition(vals,
                                kth=x+1,
                                axis=1)[:, :x+1]
        old_vals = np.sort(old_vals, axis=1)[:, 1: x+1]
        output.append(old_vals.mean(1))

    if len(output) == 1:
        output = output[0]

    return output


old20 = partial(old_n, n=20)
