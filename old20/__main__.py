"""calculate the representation distance."""
import argparse
import os

from .old20 import old_n


def write_scores(path, words, scores, overwrite):
    """Write scores to a file."""
    if os.path.exists(path) and not overwrite:
        raise ValueError("File already exists.")

    with open(path, 'w') as f:
        for word, score in zip(words, scores):
            f.write("{}\t{}\n".format(word, score))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate the old20")
    parser.add_argument("-i", "--input", type=str,
                        help="The path to the input file.",
                        required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="The path to the output file.",
                        required=True)
    parser.add_argument("-n", metavar="n", type=int, required=True)
    parser.add_argument("--overwrite",
                        const=True,
                        default=False,
                        action='store_const')
    parser.add_argument("--sep", type=str, default="\t")
    parser.add_argument("--loc", type=int, default=0)
    parser.add_argument("--header",
                        const=True,
                        default=False,
                        action='store_const')

    args = parser.parse_args()

    words = []

    for idx, x in enumerate(open(args.input).readlines()):
        if idx == 0 and args.header:
            continue
        words.append(x.strip().split(args.sep)[args.loc])

    z = old_n(words, [args.n])
    write_scores(args.output, words, z, args.overwrite)
