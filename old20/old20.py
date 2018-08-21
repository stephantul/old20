"""Calculate the OLD score for a given n."""
import numpy as np

from pyxdameraulevenshtein import damerau_levenshtein_distance
from functools import partial
from itertools import product
from tqdm import tqdm
from collections import defaultdict


def num_combinations(n, k):
    """Calculate the number of combinations."""
    return (n * (n - 1)) / k


def old_all(words, show_progressbar=True):
    """Calculate the OLD distance over all words."""
    return np.mean(old_subloop(words, show_progressbar), 1)


def old_subloop(words_a,
                words_b,
                show_progressbar,
                function=damerau_levenshtein_distance,
                n=20):
    """Calculate the distance from each word in word_a to each other word_b."""
    old_words = dict(zip(words_a, np.zeros((len(words_a), n)) + np.inf))
    maxes = dict(zip(words_a, np.zeros(len(words_a)) + np.inf))
    max_idx = defaultdict(int)
    # Levenshtein distance is symmetric
    # So we only need to calculate the distance
    # between each pair once.
    total = len(words_a) * len(words_b)
    done = set()
    for a, b in tqdm(product(words_a, words_b),
                     total=total,
                     disable=not show_progressbar):
        if a == b:
            continue
        if set((a, b)) in done:
            continue
        done.add(frozenset((a, b)))
        dist = function(a, b)
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

    return np.array([old_words[x] for x in words_a])


def old_n(words,
          reference=None,
          n=20,
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
    one_n = isinstance(n, int)
    if one_n:
        n = np.array([n])
    else:
        n = np.asarray(n)
    if np.any(n <= 0):
        raise ValueError("n should be positive.")
    if np.any([len(words) <= x for x in n]):
        raise ValueError("The number of words you have is lower than or equal "
                         "to n. Please lower n.")
    if len(set(words)) != len(words):
        raise ValueError("There are duplicates in your dataset. Please remove"
                         " these, as they will make your estimates unreliable")

    if reference is None:
        reference = words

    vals = old_subloop(words,
                       reference,
                       show_progressbar,
                       function,
                       max(n))

    vals = np.sort(vals, axis=1)

    if one_n:
        return vals[:, :n[0]].mean(1)

    return [vals[:, :x].mean(1) for x in n]


old20 = partial(old_n, n=20)
