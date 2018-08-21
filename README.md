# old20
Calculate Yarkoni, Baloto & Yap's OLD20.

* **Warning** this implementation of OLD20 uses the Damerau Levenshtein Distance instead of plain Levenshtein distance. This change has a small but non-significant effect in all experiments we ran. Nonetheless, it is important to be aware of the difference.

* **Warning** this code is _slow_ compared to the [leven_c](http://speech.ilsp.gr/iplr/downloads.htm#leven) utility. If you just need OLD20 scores, use that one.

The OLD20 distance is defined as the average Orthographic Levenshtein Distance (OLD) to the 20 closest neighbors for a given word in a given corpus.

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
from old20 import old20
# Assumes some wordlist.
wordlist = open("mywords.txt").readlines()

words_neighborhood = old20(wordlist)
```


### Author

Stéphan Tulkens
