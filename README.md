# teeny-tiny-t9

**1,381 bytes contains the entire T9 predictive text language -- 617 English words, full keyboard mapping, and a working CLI. No dependencies. No training. No data files. Just one Python script smaller than this paragraph.**

## main.py

```python
import bz2,base64,sys;unfold=lambda s=bz2.decompress(base64.b85decode(b"LRx4!F+o`-Q(4*-
7=-`<G#P+E0C)gD|KIRX1Hdts_jQ^95mWF@Gz4jhG}MA8q}nw!q}e7$Kn+chnudT=)CNL;00009Pe}w!skJo
3WC+Kd?IuV2=r{LfSdS5kp^wxbNwWS$Vi$Ny@Qii8FC_;m`x<z~L^PoymO`P*)&+L>YDER2{wz1Y?a36OnHU
db6gDFn+pZzaFnj1H-0Rj=jyouAp)gCh*MJ|BTmcoXyxE5g;>`8pU7nzY$Z|(zHwmlTlpNf^tG7UbDQ2^HQf
<iaN+a@Ng$f)PCh-%uNG`Ia<_a`{5s)PXZ7!OpS#Wd2PHwKQ4ra)9SWiv6ld%@13ppOqn=PhHrP1a{E<EcKOKG
Fk<`ebh#qk0|f{xpp28xr$TzA92U?#q>oh8UF=3z?>!y;}|SseliKaA;XA<u-`;b1)(B`)X4-zmFm@UyDRQWou
=WK;|)m%iFVkkX7rVOticoOl(Q5*ZqUEgbfAIM$+Di#_|!@2PEb&qP?g@U_u|m)hiBVk{SSp&T56ib7V@dHVoZ
TEKeseP?X<J<O@=$Qa>`D4X@9Vzg%PoKxLm@R#Yv!F<WiFu>~tDMRRY{9XupLsD3vq~|*N+=X4EoNwQ9(B9rB!
(?VgLNCWa<|^It5lq{mkx;fUk8dAC&G2ciT;1lw2xV&(TRLu=v?t{IftFJYz$|h)!HY|ga#8B`!>|e|$z?ejZo
blGW~oi>+%&+Kd?QvdrPN+|*^Nwc2FVRxELGFR>CRwmzo@nA)=LKxdNo*mu@s~Ia6G<G2cdnl^qzxWdJ$Ke;Rr
85CWH|vJF7c6E>*=a6cG#*nycXBJRyY65M6O7<3|g!uvBxLm5hfk-`9+3Oxk`tLl5#!pL8d9x)JTD4FWY=VX1t
cZcz~8$TcLyExU!>4W}FiGv;qb(z0NcnGq^5(X=HZc}UiED_4I1Zy&vL!6{a{OKaz*V^gd^*wXle3eeW;w8-C1
TvBn|`C`A9O?0O-e{5@-ywzc@w^>HMn0otQy2T0uN@h``!`5+DJu||!JDKhZ3K@73jZ@BxNqgrZ2ZWFKyOJrwgo
e<l#3%")),p=b'':[(p:=p[:d[0]&3]+d[1:])[::-1].decode()for d in s.split()];W=unfold();D={}
for w in W:k="".join(str((ord(c)-91-(c>'r')-(c>'y'))//3)for c in w);D[k]=D.get(k,[])+[w];t9=lambda*_:D
if __name__<"a":[print(*D.get(q,"?"))for q in sys.argv[1:]]or print(617,*W)
```

That's it. The entire program.

## Usage

```bash
python main.py 843              # the tie
python main.py 4663             # gone good home
python main.py 843 4663 2255    # batch lookup
python main.py                  # prints all 617 words
```

```python
from main import unfold, t9
words = unfold()                # ['able', 'about', ..., 'young']
ix = t9()                       # {'843': ['the', 'tie'], ...}
```

## How It Works

The 1,048-character blob is 838 bytes of bz2-compressed data encoded in base85.
Inside the blob: 617 English words, **reversed** and delta-encoded. Each entry is
a single digit (shared prefix length with the previous word) followed by the novel
suffix, space-separated.

Why reversed? English words share more suffixes than prefixes:

```
  -ing:  ring, king, sing, bring, thing, ...
  -er:   under, over, after, never, ...
  -ed:   used, need, based, ...
```

Reversing converts suffix overlap into prefix overlap. Delta encoding captures it.
bz2's Burrows-Wheeler Transform compresses the result 30 bytes smaller than the
forward-sorted version. The `[::-1]` in the decoder costs 5 bytes. Net: -25 bytes.

The T9 keypad mapping is computed, not stored:

```python
digit = (ord(c) - 91 - (c>'r') - (c>'y')) // 3
```

This formula handles the irregular 4-letter groups (PQRS=7, WXYZ=9) with boolean
arithmetic. No lookup table needed.

## The Journey

This repo began as a neural network research project -- eigenspace projections,
spectral initialization, CMA-ES search, transformers, rented GPUs, 40,000 lines
of code. The finding: `numpy.linalg.lstsq(X, Y)` hits 81.8%, the theoretical
ceiling. Every neural network scored lower. T9 is a lookup problem, not a
learning problem.

So we deleted the neural networks, shipped the lookup table, and then spent
28 experiments compressing it to within **0.36% of its information-theoretic
floor**. See [WHITEPAPER.md](WHITEPAPER.md) for the full research.

```
  2,101 bytes  starting point
  1,381 bytes  final  (-34.3%, 0.36% from architecture floor)
    838 bytes  compressed data (bz2, proven optimal with stdlib)
    333 bytes  code (every byte load-bearing, verified by automated scan)
```

## License

MIT
