"""Calculate the OLD score for a given n."""
import numpy as np

from pyxdameraulevenshtein import damerau_levenshtein_distance
from functools import partial
from multiprocessing import Pool, cpu_count


def calc_dist(a, words_b, n, function):
    """Sub for parallelization."""
    max_idx = 0
    max_val = np.inf
    row = np.zeros(n) + np.inf
    for b in words_b:
        dist = function(a, b)
        if max_val > dist:
            row[max_idx] = dist
            m = np.argmax(row)
            max_idx = m
            max_val = row[m]

    return (a, row)


def old_subloop(words_a,
                words_b,
                function=damerau_levenshtein_distance,
                n=20,
                n_jobs=-1):
    """Calculate distance from each word in word_a to each word in word_b."""
    func = partial(calc_dist, words_b=words_b, n=n, function=function)

    if n_jobs == -1:
        n_jobs = cpu_count()

    with Pool(n_jobs) as p:
        result = dict(p.map(func, words_a))

    return [result[x] for x in words_a]


def old_n(words,
          reference=None,
          n=20,
          show_progressbar=True,
          function=damerau_levenshtein_distance,
          n_jobs=-1):
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
    if reference is None:
        reference = words
    if one_n:
        n = np.array([n])
    else:
        n = np.asarray(n)
    if np.any(n <= 0):
        raise ValueError("n should be positive.")
    if np.any([len(reference) <= x for x in n]):
        raise ValueError("The number of reference words you have is lower than"
                         " or equal to n. Please lower n.")
    if len(set(words)) != len(words):
        raise ValueError("There are duplicates in your dataset. Please remove"
                         " these, as they will make your estimates unreliable")

    vals = old_subloop(words,
                       reference,
                       function,
                       max(n))

    vals = np.sort(vals, axis=1)

    if one_n:
        return vals[:, :n[0]].mean(1)

    return [vals[:, :x].mean(1) for x in n]


old20 = partial(old_n, n=20)
