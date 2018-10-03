# old20
Calculate Yarkoni, Baloto & Yap's OLD20.

The OLD20 distance is defined as the average Orthographic Levenshtein Distance (OLD) to the 20 closest neighbors for a given word in a given corpus.
This is a multi-threaded version of OLD20 which uses a fast, cythonized version of the Levenshtein distance, although we support any function that takes two strings and outputs a distance between them.
It can therefore also be used with the Levenshtein distances in `difflib` and `pyxdameraulevenshtein`.

* **Warning** this code is slower than the [leven_c](http://speech.ilsp.gr/iplr/downloads.htm#leven) utility. If you just need OLD20 scores, use that one.
As a comparison, `leven_c` takes about 14 seconds to process 10,000 words, while this implementation takes about 21 seconds on the same corpus.
Our main speed bottleneck is the levenshtein distance calculation, which, even though we use a fast implementation from [jellyfish](https://github.com/jamesturk/jellyfish), is still a lot slower than the Levenshtein calculation in `leven_c`.

If you use the code in this repository, please cite the following paper:

```
@article{yarkoni2008moving,
  title={Moving beyond Coltheart’s N: A new measure of orthographic similarity},
  author={Yarkoni, Tal and Balota, David and Yap, Melvin},
  journal={Psychonomic Bulletin \& Review},
  volume={15},
  number={5},
  pages={971--979},
  year={2008},
  publisher={Springer}
}
```

### Example

```python
import numpy as np
from old20 import old20, old_n
# Assumes some wordlist.
wordlist = open("mywords.txt").readlines()

words_neighborhood = old20(wordlist)

# Get multiple n
words_neighborhood = old_n(wordlist, n=[20, 30, 40, 50])

# Only calculate some words against a reference corpus.
words_subset = old_n(wordlist[:100], wordlist, n=20)

# Change the number of CPU jobs (the default is to use all cpus)
words_neighborhood = old_n(wordlist, n_jobs=2)

```

### Author

Stéphan Tulkens
