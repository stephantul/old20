# old20
Calculate Yarkoni, Baloto & Yap's OLD20.

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
from old20 import old20
# Assumes some wordlist.
wordlist = open("mywords.txt").readlines()

words_neighborhood = old20(wordlist)
```


### Author

Stéphan Tulkens
