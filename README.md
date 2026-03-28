# teeny-tiny-t9

**24 lines of Python. 1KB seed. 617 words. Full T9.**

```
$ python main.py 843 4663 2255
843  -> the, tie
4663 -> gone, good, home
2255 -> ball, call
```

## How It Works

A 1,041-byte binary seed decompresses into a 2,051-character shape string.
The shape is a prefix trie: `^` branches between groups, `]` opens a prefix,
`|` forks between suffixes. One list comprehension walks it:

```python
unfold = lambda s=S: [p+x for b in s.split("^")
    for p,_,t in [b.partition("]")] for x in t.split("|")]
```

617 words fall out. A 26-character keypad string maps them to T9:

```python
K = "22233344455566677778889999"   #  a->2, b->2, c->2, d->3, ...
```

That's the whole engine.

## T9 Keypad

```
    ┌───────┬───────┬───────┐
    │ 2 abc │ 3 def │ 4 ghi │
    ├───────┼───────┼───────┤
    │ 5 jkl │ 6 mno │ 7 pqrs│
    ├───────┼───────┼───────┤
    │ 8 tuv │ 9 wxyz│       │
    └───────┴───────┘
```

## The Numbers

```
    617 words          0 dependencies
  2,051 char shape     0 training time
  1,041 byte seed     24 lines of code
    505 T9 sequences  100% top-5 accuracy
     95 collisions   81.8% top-1 accuracy (theoretical ceiling)
```

### Why 81.8% is the maximum

505 unique digit sequences for 617 words. 410 have one word (always right).
95 share keys with other words. Best you can do: 505/617 = 81.8%.

```
  1 word  ████████████████████████████████████████░░  410
  2 words ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   81
  3 words █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   11
  4 words ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    3
```

Top-5 = **100%** because no group exceeds 4.

### Compression layers

```
  English words       2,357 bytes   ██████████████████████████████
  Readable shape      2,051 chars   ██████████████████████████░░░░
  zlib(shape)         1,041 bytes   █████████████░░░░░░░░░░░░░░░░░
  information min      ~467 bytes   ██████░░░░░░░░░░░░░░░░░░░░░░░░
```

## What We Proved

This repo started as a neural network research project: eigenspace projections,
spectral initialization, CMA-ES search, transformers, rented GPUs, 40,000 lines of code.

The finding: `numpy.linalg.lstsq(X, Y)` — one line — hits 81.8%, the theoretical
ceiling. Every neural network scored lower. T9 is a lookup problem.

So we deleted the neural networks and shipped the lookup table.

```
  np.linalg.lstsq       81.8%    ████████████████████ ceiling
  Eigenspace + MLP       74.6%    ██████████████████░░
  T9 Transformer         23.0%    █████░░░░░░░░░░░░░░░
  Xavier MLP              1.3%    ░░░░░░░░░░░░░░░░░░░░
```

## Usage

```bash
python main.py                    # print all 617 words
python main.py 843                # the, tie
python main.py 843 4663 2255      # batch lookup
```

```python
from main import unfold, t9
words = unfold()                  # 617 words
ix = t9()                         # {digits: [words]}
ix["4663"]                        # ['gone', 'good', 'home']
```

## Requirements

None.

## License

MIT
