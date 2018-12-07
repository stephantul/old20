"""Fast calculation of coltheart's N."""
import numpy as np

from jellyfish import hamming_distance
from functools import partial
from multiprocessing import Pool, cpu_count

# N is symmetric, but also takes a lot of storage.


def calc_dist(a, words_b, threshold, function):
    """Sub for parallelization."""
    scores = np.zeros(threshold)
    for b in words_b:
        dist = function(a, b)
        if 0 < dist <= threshold:
            scores[dist-1] += 1

    return (a, scores)


def n_subloop(words_a,
              words_b,
              function=hamming_distance,
              threshold=1,
              n_jobs=-1):
    """Calculate distance from each word in word_a to each word in word_b."""
    func = partial(calc_dist,
                   words_b=words_b,
                   threshold=threshold,
                   function=function)

    if n_jobs == -1:
        n_jobs = cpu_count()

    with Pool(n_jobs) as p:
        result = dict(p.map(func, words_a))

    return np.array([result[x] for x in words_a])


def n_n(words,
        reference=None,
        n=1,
        function=hamming_distance,
        n_jobs=-1):
    """
    Calculate the N distance for a given threshold.

    n can be a number, or a list of numbers. If a list of numbers is passed,
    the old_matrix is calculated and then truncated for each n.
    This is therefore vastly faster than calling old_n with a single n in
    a for loop.

    Parameters
    ----------
    words : list
        A list of strings, representing all the types in your corpus.
    n : int or list of ints

    Returns
    -------
    The N score for a given n.

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
    if np.any([len(reference) < x for x in n]):
        raise ValueError("The number of reference words you have is lower than"
                         " or equal to n. Please lower n.")
    if len(set(words)) != len(words):
        raise ValueError("There are duplicates in your dataset. Please remove"
                         " these, as they will make your estimates unreliable")

    vals = n_subloop(words,
                     reference,
                     function,
                     max(n),
                     n_jobs)

    if one_n:
        return np.sum(vals, 1)

    return np.cumsum(vals, axis=1)
