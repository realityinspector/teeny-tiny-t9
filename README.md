# teeny-tiny-t9

**1,381 bytes contains the entire T9 predictive text language — 617 English
words, full keyboard mapping, and a working CLI. No dependencies. No training.
No data files. Just one Python script smaller than this paragraph.**

```bash
python main.py 843              # the tie
python main.py 4663             # gone good home
python main.py                  # prints all 617 words
```

The 1,048-character blob is 838 bytes of bz2-compressed data encoded in base85.
Inside: 617 English words, reversed and delta-encoded. bz2's Burrows-Wheeler
Transform compresses reversed English words 30 bytes smaller than forward-sorted.
See [archive/docs/WHITEPAPER.md](archive/docs/WHITEPAPER.md) for the full
compression research (28 techniques tested).

## Prism

The neural network research in this repo grew into
**[Prism: Prismic Pretraining Acceleration](https://github.com/realityinspector/prismic-pretraining)** —
a spectral transfer method that achieves 3.33x faster GPT-2 convergence.
The Prism code and findings now live in their own repo.

## License

Apache 2.0
