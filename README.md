# teeny-tiny-t9

**3 lines of Python. 1,381 bytes. 617 words. Full T9.**

```
$ python main.py 843 4663 2255
the tie
gone good home
ball call
```

## How It Works

An 838-byte bz2-compressed seed contains 617 common English words, reversed
and delta-encoded to exploit suffix sharing. A walrus-operator list comprehension
decodes them in one expression:

```python
unfold = lambda s=BLOB, p=b'': [
    (p := p[:d[0]&3] + d[1:])[::-1].decode()
    for d in s.split()
]
```

617 words fall out. A T9 index is built at import time using an arithmetic
formula that computes the keypad mapping without storing it:

```python
# T9 digit for letter c:  str((ord(c) - 91 - (c>'r') - (c>'y')) // 3)
# Handles the irregular 4-letter groups (PQRS=7, WXYZ=9)
```

That's the whole engine.

## T9 Keypad

```
    +---------+---------+---------+
    | 2  abc  | 3  def  | 4  ghi  |
    +---------+---------+---------+
    | 5  jkl  | 6  mno  | 7 pqrs  |
    +---------+---------+---------+
    | 8  tuv  | 9 wxyz  |         |
    +---------+---------+---------+
```

## The Numbers

```
  2,101 bytes   starting point (24 lines, base64+zlib, shape format)
  1,381 bytes   final (3 lines, b85+bz2, reversed delta)
    720 bytes   cut  (34.3% reduction, 28 techniques tested)
    505 T9 sequences    100% top-5 accuracy
     95 collisions     81.8% top-1 accuracy (theoretical ceiling)
      0 dependencies
```

### Compression layers

```
  617 English words    2,886 bytes   ##############################
  Reversed delta       2,202 chars   ##########################....
  bz2(reversed delta)    838 bytes   #########.....................
  b85(bz2)             1,048 chars   ###########...................
  Architecture floor   1,376 bytes   ###############...............
  Final main.py        1,381 bytes   ###############...............  (0.36% above floor)
```

### Why reversed delta?

English words share more suffixes (-ing, -er, -ed, -tion) than prefixes.
Reversing each word before sorting converts suffix overlap into prefix
overlap, which delta encoding captures directly:

```
  Forward sorted delta:  1,181 shared prefix chars  ->  bz2 = 868 bytes
  Reversed sorted delta: 1,301 shared prefix chars  ->  bz2 = 838 bytes
                         +10.2% more sharing         ->  -30 bytes
```

### Why 81.8% is the maximum

505 unique digit sequences for 617 words. 410 have one word (always right).
95 share keys with other words. Best you can do: 505/617 = 81.8%.
Top-5 = **100%** because no group exceeds 4.

## What We Proved

This repo started as a neural network research project: eigenspace projections,
spectral initialization, CMA-ES search, transformers, rented GPUs, 40,000 lines
of code. The finding: `numpy.linalg.lstsq(X, Y)` hits the theoretical ceiling.
Every neural network scored lower. T9 is a lookup problem.

So we deleted the neural networks and shipped the lookup table. Then we
compressed it to within 0.36% of its information-theoretic floor.

See [WHITEPAPER.md](WHITEPAPER.md) for the full hypercompression research.

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

Python 3.8+ (walrus operator). No dependencies.

## License

MIT
