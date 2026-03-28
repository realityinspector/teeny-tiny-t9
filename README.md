# teeny-tiny-t9

**2KB of Python that contains 617 English words and a complete T9 predictive text engine.**

No dependencies. No training. No neural network. Just a shape that unfolds.

```
$ python main.py 843 4663 2255
843  -> the, tie
4663 -> gone, good, home
2255 -> ball, call
```

## The Shape

A 2,051-character string encodes the 617 most common English words
as a compressed prefix trie. A one-line function unfolds it:

```python
S = "a]ble|ct|dd|ge|go|ir|ll|lso|m|n|nd|ny|rea|rm|rmy|rt|sk|t|te|way^ba]ck|d|g|ll..."

unfold = lambda s=S: [p+x for b in s.split("^") for p,_,t in [b.partition("]")] for x in t.split("|")]

>>> unfold()
['able', 'act', 'add', 'age', 'ago', 'air', 'all', 'also', ... ]  # 617 words
```

The `^` branches between prefix groups. The `]` opens a prefix. The `|` forks between suffixes.
Every word in the dictionary grows from one walk through this shape.

## How T9 Works

T9 maps phone keys to letters. Multiple words share the same key sequence:

```
    ┌─────┬─────┬─────┐
    │ 2   │ 3   │ 4   │
    │ abc │ def │ ghi │
    ├─────┼─────┼─────┤
    │ 5   │ 6   │ 7   │
    │ jkl │ mno │pqrs │
    ├─────┼─────┼─────┤
    │ 8   │ 9   │     │
    │ tuv │wxyz │     │
    └─────┴─────┘     │
                      │
    617 words  ───────┘
    505 key sequences
     95 collisions
```

## The Numbers

```
   617 words
 2,051 characters of shape data
    19 lines of logic
     0 dependencies
     0 training time
   100% top-5 accuracy
  81.8% top-1 accuracy (theoretical ceiling)
```

### Why 81.8% is the ceiling

505 unique key sequences map to 617 words. 410 sequences have exactly
one word — always correct. 95 sequences are ambiguous (multiple words
share the same keys). You can only guess one, so at most 505/617 = 81.8%.

```
  Ambiguity breakdown:

  1 word  ████████████████████████████████████████░░  410 sequences
  2 words ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   81 sequences
  3 words █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   11 sequences
  4 words ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    3 sequences
           0        100       200       300       400     500
```

The top-5 accuracy is **100%** because no group has more than 4 words.

### Famous collisions

```
  843  -> the, tie           4653 -> gold, golf, hold, hole
  4663 -> gone, good, home   2273 -> base, card, care, case
  729  -> pay, raw, saw, say  227 -> bar, cap, car
```

### Word length distribution

```
  2 ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  23
  3 ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 152
  4 ████████████████████████████████████████░ 442
```

## What We Proved Along the Way

This repo started as a research project asking: *can a neural network beat
a lookup table at T9?*

We tried everything:
- Eigenspace projection (SVD of input-output correlation)
- Spectral weight initialization (CMA-ES search over singular value shapes)
- Task-aligned anisotropy (rotating eigenvectors toward class centroids)
- A custom transformer architecture (T9T)
- 30+ automated experiments on rented GPUs

The answer: **no.** `numpy.linalg.lstsq(X, Y)` — a single line — achieves
81.8%, the theoretical maximum. Every neural network we trained scored lower.

```
  Approach                   Accuracy   vs Ceiling
  ─────────────────────────  ────────   ──────────
  np.linalg.lstsq (1 line)    81.8%    ████████████████████ 100%
  Eigenspace + 2-layer MLP    74.6%    ██████████████████░░  91%
  T9T Transformer (50 ep)     23.0%    █████░░░░░░░░░░░░░░░  28%
  Xavier MLP (50 ep)            1.3%    ░░░░░░░░░░░░░░░░░░░░   2%
```

T9 is a lookup problem, not a learning problem. The optimal solution
is a table, not a function. So we built the table — and compressed it
into a shape.

## Usage

```bash
python main.py                         # print all 617 words
python main.py 843                     # T9 lookup: "the, tie"
python main.py 843 4663 2255 5878      # batch lookup
```

```python
from main import unfold, t9

words = unfold()          # 617 words from the shape
ix = t9()                 # {digit_string: [words]}
ix["4663"]                # ['gone', 'good', 'home']
```

## Requirements

None. Pure Python, stdlib only.

## License

MIT
