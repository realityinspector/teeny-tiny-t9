# hypercompression.md

## Goal

Minimize `main.py` to the smallest file that correctly unfolds 617 English
words and solves T9 — while keeping the code clear enough that a reader
understands what's happening. Smallest total bytes wins. Ties broken by
fewer lines, then by readability.

## Current State

```
main.py   2,101 bytes   24 lines   1,041-byte zlib blob + decoder
```

Baseline metric: **2,101 bytes total file size.**

Output requirements (the test harness):

```bash
python main.py              # prints all 617 words
python main.py 843           # prints "843 -> the, tie"
python main.py 843 4663 2255 # batch mode
```

```python
from main import unfold, t9
assert len(unfold()) == 617
assert "the" in t9()["843"]
```

All four must pass for a commit to be kept.

## The Metric

**Primary: `wc -c main.py`** (total bytes). Lower is better.

**Secondary: line count.** Fewer lines at same byte count wins.

**Hard constraint:** The test harness above must pass. If it fails, the
experiment is a crash — revert.

## Architecture of the Problem

The file has three components. Each is a compression target:

```
  COMPONENT          CURRENT     THEORETICAL MIN    NOTES
  ─────────────────  ──────────  ─────────────────  ──────────────
  Data blob          1,388 chars   ~700 chars (b85)  the word list
  Decoder logic        315 chars   ~200 chars        unfold + t9
  Framing (imports,    398 chars   ~150 chars        shebang, main
   shebang, __main__)
  ─────────────────  ──────────  ─────────────────  ──────────────
  TOTAL              2,101 bytes   ~1,050 bytes
```

### Data blob — where the bytes are

The 617 words compress to 1,041 bytes via zlib. The blob is stored as
base64 in the source, expanding it to 1,388 chars. Opportunities:

1. **base85 instead of base64.** `base64` encodes 3 bytes → 4 chars (33%
   overhead). `base85` encodes 4 bytes → 5 chars (25% overhead). Saves
   ~110 chars on the blob.

2. **Different pre-compression representations.** The shape string
   (`a]ble|ct|dd|...`) compresses better than raw comma-separated words
   because it shares prefixes. But other representations may compress
   better still:
   - Sorted words with delta encoding (shared prefix length + suffix)
   - T9-digit-sorted words (groups collisions, better locality for zlib)
   - Two-stream: T9 digit trie + letter-choice bitstream
   - Custom alphabet reduction (only 28 unique chars in the shape)

3. **Skip base-encoding entirely.** Store raw bytes in a Python byte
   literal using `\x` escapes. Eliminates the base64/85 import and
   encoding overhead entirely. A 1,041-byte zlib blob as `b"\x78\x9c..."`
   costs ~4× in hex escapes (4,164 chars) — worse. But with latin-1
   passthrough for printable bytes, many stay as single chars.

4. **Encode into the shape string format directly.** Skip zlib. The shape
   string is 2,051 chars but needs no imports (no `zlib`, no `base64`).
   Trade: bigger data, smaller decoder. The crossover point determines
   which wins.

### Decoder logic — where the cleverness is

Current `unfold` is a one-liner list comprehension (96 chars). Current `t9`
is 5 lines. Opportunities:

1. **Combine unfold + t9 into one function.** Build the T9 index during
   the unfold walk, returning both the word list and the index.

2. **Inline the keypad.** `"22233344455566677778889999"` is 26 chars.
   Could compute it: `"".join(str(i//3+2) for i in range(26))` — but
   that's wrong for 7/9 (4 letters each). The string is already minimal.

3. **Shorten variable names.** `unfold` → `u`, `t9` → `t`. Saves a few
   bytes but hurts readability. Only do this near the end.

4. **Lambda vs def.** `unfold` is already a lambda. `t9` could be too,
   but the loop body needs `setdefault` which is awkward in a lambda.

### Framing — where convention costs bytes

1. **Drop the shebang.** `#!/usr/bin/env python3` is 22 bytes. It's
   convention, not function.

2. **Drop the docstring.** 73 bytes.

3. **Minimize `__main__`.** Current is 5 lines. Could be 2 with
   semicolons.

4. **Drop the `if __name__` guard.** If we accept that importing the
   module prints nothing (it doesn't — the guard handles that), we
   could inline the CLI at module level behind `import sys;
   sys.argv[1:]and...`. Risky and ugly.

## The Experiment Loop

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):

### Setup (once)

```bash
cd /path/to/teeny-tiny-t9
git checkout -b autoresearch/hypercompress
cp main.py main.py.baseline
echo "commit\tbytes\tlines\tstatus\tdescription" > results.tsv
```

### The loop (runs until stopped)

```
LOOP FOREVER:
  1. Read main.py. Note current byte count.
  2. Form a hypothesis: "X will reduce bytes because Y."
  3. Edit main.py.
  4. Run the test harness:

     python -c "
     from main import unfold, t9
     W = unfold()
     assert len(W) == 617 and len(set(W)) == 617
     K = '22233344455566677778889999'
     ix = t9(W)
     assert len(ix) == 505
     assert 'the' in ix['843']
     for w in W:
         d = ''.join(K[ord(c)-97] for c in w)
         assert w in ix[d]
     import subprocess
     r = subprocess.run(['python', 'main.py', '843'], capture_output=True, text=True)
     assert 'the' in r.stdout
     r2 = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
     assert '617' in r2.stdout
     import os
     print(f'PASS: {os.path.getsize(\"main.py\")} bytes')
     "

  5. If test fails → fix trivially or revert. Log as "crash".
  6. Compare byte count to previous best.
  7. If smaller → git commit, log as "keep".
     If equal but fewer lines → git commit, log as "keep".
     If equal but cleaner → git commit, log as "keep (readability)".
     If larger → git checkout main.py, log as "discard".
  8. Append to results.tsv:
     {commit}\t{bytes}\t{lines}\t{status}\t{description}
  9. Go to 1.
```

### Scoring

```
SCORE = file_bytes

KEEP if:
  score < previous_best
  OR (score == previous_best AND lines < previous_lines)
  OR (score == previous_best AND lines == previous_lines AND simpler)

DISCARD otherwise.
```

## Experiment Queue

Ordered by expected impact. Run these first.

### Phase 1: Low-hanging fruit (est. savings: 200-400 bytes)

| #  | Experiment | Hypothesis | Est. |
|----|-----------|-----------|------|
| 1  | base85 encoding | 25% overhead vs 33%. Saves ~110 chars on blob, minus `a85decode` vs `b64decode` (same length). Net ~100 bytes. | -100 |
| 2  | Drop shebang + docstring | 95 bytes of ceremony. | -95 |
| 3  | Minimize `__main__` block | Compress to 2 lines with semicolons. | -40 |
| 4  | Raw byte literal | Skip base64/85 entirely. Encode 1,041 zlib bytes as `b"..."` with `\x` escapes only for non-printable bytes. Saves import line + encoding overhead. | -50 to +200 |

### Phase 2: Re-encode the data (est. savings: 100-300 bytes)

| #  | Experiment | Hypothesis | Est. |
|----|-----------|-----------|------|
| 5  | Compress sorted words directly | Skip the shape format. `zlib(sorted_words)` may compress differently than `zlib(shape)`. Test both. | ? |
| 6  | T9-sorted word order | Words sorted by T9 digit sequence group collisions together. May compress better. | ? |
| 7  | Custom 5-bit alphabet | Shape uses 28 chars. Pack 5 bits/char → 1,282 bytes before zlib. May give zlib better byte-level patterns. | ? |
| 8  | Delta-encode the shape | Adjacent suffixes often share characters. Delta encoding before zlib. | ? |
| 9  | Two-stream encoding | Separate T9 trie topology from letter choices. Trie is ~176 bytes succinct. Letter choices are ~185 bytes of entropy. Total ~361 bytes raw. | -200 (raw) but decoder adds ~300 |
| 10 | Reorder shape blocks | zlib uses LZ77 (sliding window). Reordering prefix groups to maximize back-references. | ? |

### Phase 3: Shrink the decoder (est. savings: 50-150 bytes)

| #  | Experiment | Hypothesis | Est. |
|----|-----------|-----------|------|
| 11 | Merge unfold + t9 | Single function returns both word list and T9 index. Eliminates one `def` and one iteration over words. | -30 |
| 12 | Inline everything | Replace `unfold` and `t9` with direct expressions in `__main__`. Loses importability but removes function overhead. | -40 |
| 13 | Shorter variable names | `S→s`, `unfold→u`, `t9→t`. | -20 |
| 14 | Exploit `exec`/`eval` | Compress the decoder code itself and eval it. Meta-compression. Only if the code savings exceed the `exec()` overhead. | ? |

### Phase 4: Radical restructuring (est. savings: 0-500 bytes)

| #  | Experiment | Hypothesis | Est. |
|----|-----------|-----------|------|
| 15 | No zlib — raw shape string | Skip compression entirely. The shape string is 2,051 chars. With no imports, no decode step, just `S="..."` and the one-liner unfold: total ~2,300 bytes. Currently at 2,101. Loses if shape > ~1,700 chars, wins if decoder overhead exceeds ~400 bytes. | +200 (lose) |
| 16 | Polyglot encoding | Encode the word list in a format that serves double duty: both the data AND the T9 index. Each word stored with its T9 key inline. | ? |
| 17 | Generative model | Instead of storing words, store a T9 trie + letter-choice bitstream. The trie determines which digit sequences are valid. The bitstream picks which letter each digit means. ~446 bytes of data but ~300 bytes of decoder. | ~750 total? |
| 18 | Self-extracting zip | `python -c "import zipfile;..."` or similar. Python can import from zip files. Package main.py inside itself. Cursed but possibly smallest. | ? |

## Information-Theoretic Floor

```
  617 words, avg 3.8 letters, 28 unique chars in shape
  Shannon entropy: ~467 bytes (conditional on T9 structure)
  zlib achieves:   1,041 bytes (2.2x theoretical)
  Decoder minimum:  ~150 bytes (imports + unfold + t9 + main)
  ─────────────────────────────────────────────────────────
  Theoretical minimum file: ~617 bytes
  Realistic minimum file:  ~900-1,100 bytes
```

Below ~900 bytes, the decoder overhead exceeds the data savings from
exotic encodings. The sweet spot is probably **1,100-1,400 bytes** with
clean, readable code.

## What "Clean Demonstration" Means

The file should satisfy a reader who opens it on GitHub:

1. **They can tell what it does** without running it (docstring or clear
   variable names).
2. **They can see the shape** — even compressed, a comment like
   `# 1KB seed -> 617 English words` tells the story.
3. **The CLI works** — `python main.py 843` should feel like using T9.
4. **The import works** — `from main import unfold, t9` should be
   clean API.

Trade readability for bytes only after exhausting all encoding tricks.
Renaming `unfold` to `u` is a last resort, not a first move.

## Running the Loop

Local CPU only. Each experiment takes <1 second (no training — just
file encoding and test harness). Expected throughput: ~120 experiments
per hour with an autonomous agent, limited by hypothesis generation.

```bash
# Run the test harness (the only command the loop needs)
python -c "
from main import unfold, t9
W=unfold(); ix=t9(W)
assert len(W)==617 and len(set(W))==617 and len(ix)==505
assert 'the' in ix['843'] and 'good' in ix['4663']
import os; print(f'PASS: {os.path.getsize(\"main.py\")} bytes, {sum(1 for _ in open(\"main.py\"))} lines')
"
```

Time budget per experiment: **5 seconds** (generous — most take <0.1s).

## Endgame

When the loop plateaus (3+ consecutive discards with no new ideas),
switch to the readability pass: at the current byte count, make the
code as clear as possible. This is a free optimization — same score,
better file.

The dream target:

```
main.py   ~1,100 bytes   ~15 lines   full T9   zero dependencies   readable
```
