# signal_synthesis.md

## The Question

Can a formula or equation *generate* the T9 word seed — bypassing storage
entirely? Instead of compressing data, could we collapse it into a
parametric signal: a set of oscillator coefficients, a chaotic attractor's
initial conditions, or a generative function that, when evaluated, produces
the exact byte sequence?

## The Information-Theoretic Wall

Before experimenting, we must name the constraint we're fighting.

The 617-word T9 list compresses to **868 bytes** under bz2. This is
approximately its **Kolmogorov complexity** — the shortest program that
produces it. Shannon's theorem guarantees:

```
  Any exact reconstruction scheme must encode ≥ 6,944 bits of information.
  This holds regardless of representation: Fourier, wavelet, polynomial,
  chaotic map, neural network, or alien mathematics we haven't invented.
```

This means: the *parameters* of any formula that generates our data must
total at least 868 bytes. The formula's *structure* (the code that
interprets those parameters) is overhead on top of that.

**The current approach (bz2 blob + decoder) is already near-optimal:**
868 bytes of data + ~50 bytes of decoder = ~918 bytes total.

Any signal-synthesis approach must beat 918 bytes total (formula code +
parameters), which is extremely tight.

## Why This Is Still Worth Exploring

The information-theoretic wall applies to *arbitrary* compressed data.
But our data isn't arbitrary — it's **617 English words**. English has
deep structure:

1. **Phonotactic constraints**: Not all letter combinations occur. "th",
   "ing", "tion" are common; "xzq" is impossible.
2. **T9 collision structure**: Words sharing a T9 key share digit patterns,
   creating algebraic relationships between words.
3. **Zipfian frequency distribution**: Common words are short; rare words
   are long.
4. **Morphological patterns**: Prefixes (un-, re-), suffixes (-ing, -ed,
   -tion), stems.

bz2 exploits *byte-level* statistical patterns via BWT + Huffman. But it
doesn't know English. A formula that encodes **linguistic structure**
directly could theoretically match bz2's byte count while also encoding
the decoder logic *inside the formula itself*.

The real question isn't "can we beat 868 bytes of parameters?" (almost
certainly not). It's: **can the formula structure replace enough decoder
code that (formula + parameters) < (bz2 blob + decoder)?**

## Approach 1: Fourier Synthesis of the Byte Stream

### Theory

Treat the 868-byte bz2 blob (or the raw word list) as a discrete signal
x[0..N-1]. The DFT gives N complex coefficients. Truncating to K < N
coefficients gives lossy approximation.

### Why it fails for compressed data

bz2 output is pseudorandom — energy is spread uniformly across all
frequencies. Truncating any coefficients corrupts the output. You need
all N coefficients (= N real parameters = same size as the data).

### Why it might work for the raw word list

The *uncompressed* word list (2,322 chars of delta-encoded text) has
strong structure: mostly lowercase letters, frequent commas, digits 0-3.
Its frequency spectrum likely has energy concentration at low frequencies
(smooth letter distributions) with sparse high-frequency components.

**Experiment**: Compute DCT of the raw delta string. Measure how many
coefficients carry 99.9%, 99.99%, 99.999% of the energy. If K << N
coefficients suffice for near-exact reconstruction, this is viable.

### Decoder cost

```python
import numpy as np
def unfold(C=COEFFICIENTS):
    x = np.fft.idct(C)
    s = bytes(round(v) % 256 for v in x).decode()
    # ... delta decode
```

Problem: requires `numpy` (not stdlib). With pure Python DCT, decoder
is ~200 bytes. Plus coefficient storage. Likely loses.

## Approach 2: Chaotic Oscillator with Exact Initial Conditions

### Theory

A chaotic map like the logistic map x_{n+1} = r·x_n·(1-x_n) produces
deterministic output from initial conditions (r, x_0). The output
*looks* random but is fully specified by two numbers.

### The precision trap

To control N output bytes, x_0 must be specified to ~N bytes of
precision (Lyapunov exponent ≈ ln(2) → 1 bit of divergence per step).

For 868 output bytes: x_0 needs ~868 bytes of precision.
**The initial condition IS the data, wearing a costume.**

### Interesting variant: coupled oscillators

What if multiple shorter-precision oscillators are combined?

```
x_n = f(osc1(n), osc2(n), ..., oscK(n))
```

Each oscillator has modest precision (say 32 bits), and the combination
function f selects the right byte at each position.

This is equivalent to a lookup table with extra steps — but the
*structure* of f might be compressible if the word list has patterns
that align with oscillator periodicities.

**Experiment**: Search for small K (4-8 oscillators) where the
combination function f is simple enough that (oscillator params +
f code) < 868 bytes. Use T9 digit structure as a guide — words
sharing a T9 key might map to nearby oscillator phases.

## Approach 3: Arithmetic on the Word Space

### Theory

Instead of encoding bytes, encode *decisions*. The T9 word list can be
seen as a sequence of choices:

1. How many words start with each letter? (26 values)
2. For each prefix, what letters follow? (branching tree)
3. Where does each word end? (binary choices)

This is essentially **arithmetic coding with a language model**. The
"formula" is the language model, and the "parameters" are the coded
bitstream.

### The key insight

If the language model is good (assigns high probability to the actual
words), the coded bitstream is short. The total cost is:

```
Total = model_size + coded_bitstream
```

A perfect model of English would make the bitstream nearly zero — but
the model itself would be enormous. The sweet spot is a tiny model
that captures enough structure to significantly shrink the bitstream.

### Candidate models

| Model | Approx. size | Expected bitstream | Total |
|-------|-------------|-------------------|-------|
| Uniform (no model) | 0 bytes | ~868 bytes | ~868 |
| Letter frequencies | 26 bytes | ~780 bytes | ~806 |
| Bigram frequencies | 676 bytes | ~600 bytes | ~1,276 |
| T9-aware trie | ~200 bytes | ~250 bytes | ~450 |
| Phonotactic FSM | ~300 bytes | ~200 bytes | ~500 |

The T9-aware trie approach is interesting: store only the trie topology
(which T9 keys have valid words) in ~176 bytes, then for each leaf,
store which letter each digit represents (the "disambiguation" bits).

**Experiment**: Build the T9 trie for 617 words. Measure:
- Trie topology size (succinct representation)
- Disambiguation bitstream size
- Decoder code size
- Compare total to current 1,454 bytes

## Approach 4: Neural Oscillator (Tiny Network as Formula)

### Theory

Train a neural network where:
- Input: position index n (0-2321, one per character in delta string)
- Output: character at position n
- Architecture: small MLP with periodic activations (sin/cos → "neural oscillator")

The weights ARE the formula parameters. The network IS the oscillator.

### SIREN architecture

Sitzmann et al. (2020) showed that MLPs with sinusoidal activations
(SIREN) can fit complex signals with very few parameters. The key
property: sin activations create implicit Fourier features, enabling
the network to represent high-frequency detail.

```python
# A tiny SIREN
def f(t, W1, b1, W2, b2, W3, b3):
    h = sin(W1 * t + b1)       # hidden layer 1
    h = sin(W2 @ h + b2)       # hidden layer 2
    return W3 @ h + b3          # output layer
```

### Parameter budget

For 2,322 output characters (the delta string):
- Input: 1 → Hidden: H → Hidden: H → Output: 1
- Parameters: H + H + H² + H + H + 1 = H² + 3H + 1
- At H=48: 48² + 3(48) + 1 = 2,449 parameters

At float32 (4 bytes): 9,796 bytes. Way too large.
At int8 (1 byte): 2,449 bytes. Still larger than current.
At 4-bit quantization: 1,225 bytes. Smaller than 1,454!

But: the decoder (matrix multiply + sin) in Python is ~300 bytes.
Total: 1,225 + 300 = 1,525. **Worse** than current 1,454.

**However**: if we target the *pre-compression* word list and combine
with a better quantization scheme, this might cross over.

**Experiment**: Train a SIREN with H=32 to fit the delta string. Measure
exact reconstruction accuracy. Find minimum H for 100% accuracy. Compute
total file size with 4-bit and 2-bit weight quantization.

## Approach 5: Algebraic Number Theory — Words as Primes

### Theory (speculative)

Map each of the 617 words to a unique prime number. The product of all
617 primes is a single large integer. Given the integer and the prime
mapping, you can recover all words via factorization.

The product of 617 small primes ≈ e^(617 × avg(ln(p))). With primes up
to ~4,600 (the 617th prime), avg(ln(p)) ≈ 7.5, so the product has
617 × 7.5 / ln(10) ≈ 2,010 decimal digits ≈ 837 bytes as a big integer.

But you also need the prime → word mapping (~1,200 bytes) and the
decoder (~100 bytes). Total: ~2,137 bytes. Worse.

### Variant: Chinese Remainder Theorem encoding

Encode each word's index using CRT with coprime moduli. This is
essentially a number-theoretic transform — same information-theoretic
limits, different algebra.

## Approach 6: Compression of the Formula Itself

### Meta-insight

What if we compress *the Python code* of a parametric generator? The
generator source code (formula + embedded parameters) might compress
better than raw data because the formula has repetitive structure
(function calls, loops, operators).

```python
exec(bz2.decompress(b"<compressed Python source that contains formula>"))
```

We already tested this in the hypercompression work — exec(compressed
source) was 2,265+ bytes vs our 1,454. The formula code compresses
poorly because it's short and non-repetitive.

## The Experiment Loop

### Setup

```bash
cd /path/to/teeny-tiny-t9
git checkout -b autoresearch/signal-synthesis
echo "exp\tbytes\tlines\tstatus\tdescription" > signal_results.tsv
```

### Phase 1: Spectral Analysis (est. 1-2 hours)

| # | Experiment | Question |
|---|-----------|----------|
| 1 | DCT of delta string | How many coefficients for exact reconstruction? |
| 2 | DCT of raw word list | Different basis, different compaction? |
| 3 | Wavelet (Haar) of delta string | Better localization than DCT? |
| 4 | Energy spectrum analysis | Where is the information concentrated? |

### Phase 2: Generative Models (est. 2-4 hours)

| # | Experiment | Question |
|---|-----------|----------|
| 5 | T9 trie topology | How many bytes for the trie structure? |
| 6 | Disambiguation bitstream | How many bits to pick letters from T9 keys? |
| 7 | Trie + bitstream vs bz2 blob | Is two-stream smaller total? |
| 8 | Letter-frequency language model | Can a 26-byte model shrink the bitstream? |
| 9 | Bigram model | Diminishing returns? |

### Phase 3: Neural Oscillator (est. 2-4 hours)

| # | Experiment | Question |
|---|-----------|----------|
| 10 | SIREN fit to delta string | Min hidden size for 100% accuracy? |
| 11 | Weight quantization sweep | 8-bit? 4-bit? 2-bit? |
| 12 | Pure-Python SIREN decoder | How small can the decoder be? |
| 13 | Total size comparison | Formula + weights vs bz2 blob? |

### Phase 4: Exotic Generators (est. 1-2 hours)

| # | Experiment | Question |
|---|-----------|----------|
| 14 | Logistic map search | Can we find (r, x0) for a partial match? |
| 15 | Coupled LFSR search | Multiple short LFSRs → target bytes? |
| 16 | Cellular automaton search | Rule + IC → target prefix? |

### Scoring

```
SCORE = total_file_bytes   (formula code + parameters + CLI)

KEEP if score < 1,454      (current best)
DISCARD otherwise, but LOG the result for theory
```

## Information-Theoretic Scoreboard

Track where we are relative to theoretical limits:

```
  BOUND                        BYTES    STATUS
  ─────────────────────────    ─────    ─────────────────
  Shannon entropy floor          868    Unreachable (no decoder)
  Kolmogorov complexity        ~900    Unreachable (optimal program)
  Current best (bz2+decoder)  1,454    ← WE ARE HERE
  Theoretical sweet spot      1,100    Goal (if structure exploitable)
  Fourier (float32)           3,572    Way worse
  Raw word list (no compress) 2,886    Way worse
```

The gap between 868 (theoretical) and 1,454 (actual) is **586 bytes**.
This is split between:
- Decoder overhead: ~370 bytes (unfold + t9 + CLI + imports)
- Encoding overhead: ~216 bytes (b85 expansion of bz2: 1085 - 868)

To beat 1,454, we'd need to find a representation where
(generator code + parameters) < 1,454. The generator code replaces
both the decoder AND the encoding overhead, so it has a budget of
~586 bytes for its own logic.

## The Deepest Question

The user's intuition — "a formula collapsed into essence" — touches
something real in algorithmic information theory. Every computable
string has a shortest program that produces it. That program IS the
string's essence, its Kolmogorov complexity.

For our T9 word list, the "essence" is approximately:

> **The 617 most common English words that fit on a T9 keypad,
> delta-encoded and bz2-compressed.**

That description is ~80 bytes of English but needs a *much* larger
program to actually execute (an English-understanding AI + bz2 +
delta-decoder). The gap between the description and the executable
program is the gap between human understanding and machine computation.

A parametric formula (oscillator, neural net, chaotic map) is a
*different computational model* for the same string. The question is
whether that model's overhead is less than Python's bz2 + delta decoder.
For our 868-byte target, the answer is almost certainly no — but for
larger word lists (10K, 100K words), the crossover might favor a
generative approach where the *model itself compresses* while the
per-word information stays constant.

## Running the Loop

```bash
# Test harness (same as hypercompression)
python3 -c "
from main import unfold, t9
W=unfold(); ix=t9(W)
assert len(W)==617 and len(set(W))==617 and len(ix)==505
assert 'the' in ix['843'] and 'good' in ix['4663']
import os; print(f'PASS: {os.path.getsize(\"main.py\")} bytes')
"
```

Time budget: **5 seconds per experiment** (some may need training loops,
budget 60 seconds for neural experiments).

The most promising path is likely **Approach 3** (T9 trie + disambiguation
bitstream), as it exploits problem-specific structure that bz2 cannot see.
