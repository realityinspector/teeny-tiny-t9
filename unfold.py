#!/usr/bin/env python3
"""
unfold.py — A shape that unfolds to English.

A 2,317-character string encodes the 617 most common English words.
An iterator walks it, branching at each ^ and | to produce words.

    python3 unfold.py           # unfold all 617 words
    python3 unfold.py --tree    # show the branching structure
    python3 unfold.py --solve   # T9 solver: keys → words, no training needed
"""

# ═══════════════════════════════════════════════════════════════════
# THE SHAPE — 617 words compressed to their branching skeleton
# Format: letter]suffix|suffix|...^letter]suffix|...
# Each ^ separates a first-letter group. Each | separates a suffix.
# ═══════════════════════════════════════════════════════════════════

S = (
    "a]ble|ct|dd|ge|go|ir|ll|lso|m|n|nd|ny|rea|rm|rmy|rt|sk|t|te|way^b]ack|ad"
    "|ag|all|and|ank|ar|ase|ath|e|ear|eat|ed|een|ell|est|ig|ill|ird|it|low|lu"
    "e|oat|ody|omb|one|ook|orn|oth|ox|oy|ug|urn|us|usy|ut|uy|y^c]all|alm|ame|"
    "amp|an|ap|ar|ard|are|ase|ast|at|ell|hin|ity|lub|oat|ode|old|ome|ook|ool|"
    "opy|ore|ost|rew|rop|up|ut^d]ad|ark|ata|ate|ay|ead|eal|ear|eep|id|ie|ig|o"
    "|og|oor|own|raw|rop|rug|ry|ust|uty^e]ach|ar|arn|ast|at|dge|gg|lse|nd|ven"
    "|ver|vil|ye^f]ace|act|ail|air|all|an|ar|arm|ast|at|ate|ear|eed|eel|ell|e"
    "w|ile|ill|ilm|ind|ine|ire|irm|ish|it|ive|ix|lat|low|ly|ood|oot|or|orm|ou"
    "r|ree|rom|uel|ull|un|und^g]ain|ame|as|ave|et|ift|irl|ive|lad|o|oal|od|oe"
    "s|old|olf|one|ood|ot|rab|ray|rew|row|ulf|un|uy^h]air|alf|all|and|ang|"
    "ad|ard|arm|as|at|ate|ave|e|ead|ear|eat|elp|er|ere|ero|ide|igh|ill|im|is|it|"
    "old|ole|oly|ome|ope|ost|ot|our|ow|uge|ung|urt^i]ce|dea|f|ll|n|ron|s|t|te"
    "m|ts^j]ack|ob|oin|ump|ury|ust^k]een|eep|ept|ey|ick|id|ill|ind|ing|nee|ne"
    "w|now^l]ack|ady|aid|ake|and|ane|ast|ate|aw|ay|ead|eft|eg|ess|et|ie|ife|i"
    "ft|ike|ine|ink|ip|ist|ive|ock|ong|ook|ord|ose|oss|ost|ot|ove|ow|uck^m]ad"
    "|ade|ail|ain|ake|ale|an|ap|ark|ass|ay|e|eal|ean|eet|et|ile|ind|ine|iss|i"
    "x|ode|om|ood|oon|ore|ost|ove|uch|ud|ust|y^n]ame|ear|eck|eed|et|ew|ews|ex"
    "t|ice|ine|o|one|or|ose|ot|ote|ow|ut^o]dd|dds|f|ff|il|ld|n|nce|ne|nly|nto"
    "|pen|r|ur|ut|ver|wn^p]ace|ack|age|aid|ain|air|ale|alm|ark|art|ass|ast|at"
    "h|ay|eak|en|er|et|ick|ie|ile|in|ine|ink|lan|lay|lot|lus|oem|oet|oll|ool|"
    "oor|op|ort|ost|ot|our|ray|ull|ure|ush|ut^r]ace|ain|an|ank|are|ate|aw|ead"
    "|eal|ear|ed|ely|est|ice|ich|id|ide|ing|ise|isk|oad|ock|ole|oll|oof|oom|o"
    "ot|ope|ose|ow|ule|un|ush^s]ad|afe|aid|ake|ale|alt|ame|and|ang|at|ave|aw|"
    "ay|ea|eat|eed|eek|eem|een|elf|ell|end|ent|et|he|hip|hop|hot|how|hut|ick|"
    "ide|ign|ing|ink|ir|it|ite|ix|ize|kin|ky|lip|low|now|o|oft|oil|old|ole|om"
    "e|on|ong|oon|ort|oul|pin|pot|tar|tay|tep|top|uch|uit|un|ure|wim^t]ail|ak"
    "e|ale|alk|all|ank|ape|ask|eam|ell|en|end|erm|est|ext|han|hat|he|hem|hen|"
    "hey|hin|his|hus|ie|ill|ime|iny|ip|ire|o|oe|old|one|oo|ook|ool|op|ops|orn"
    "|our|own|ree|rip|rue|ry|urn|win|wo|ype^u]nit|p|pon|s|se|sed|ser^v]an|ast"
    "|ery|ice|iew|ote^w]age|ait|ake|alk|all|ant|ar|arm|arn|as|ash|ave|ay|e|ea"
    "k|ear|eek|ell|ent|ere|est|et|hat|hen|ho|hom|hy|ide|ife|ild|ill|in|ind|in"
    "e|ing|ire|ise|ish|ith|on|ood|ord|ore|ork|orm|orn|rap^y]ard|eah|ear|es|et"
    "|ou|our^z]one"
)


def unfold(shape=S):
    """Unfold the shape into words. 6 lines that produce 617 words."""
    words = []
    for block in shape.split("^"):
        head, _, tail = block.partition("]")
        for suffix in tail.split("|"):
            words.append(head + suffix)
    return words


# ═══════════════════════════════════════════════════════════════════
# T9 SOLVER — maps key presses to words, zero training
#
# The T9 keypad maps letters to digits 2-9. Multiple words can share
# the same key sequence. The solver finds all matches instantly.
#
# Theoretical accuracy: 81.8% top-1 (505 unique sequences for 617
# words; 95 groups are ambiguous). Top-5 accuracy: 100%.
# ═══════════════════════════════════════════════════════════════════

KEYPAD = {
    'a':2,'b':2,'c':2, 'd':3,'e':3,'f':3, 'g':4,'h':4,'i':4,
    'j':5,'k':5,'l':5, 'm':6,'n':6,'o':6, 'p':7,'q':7,'r':7,'s':7,
    't':8,'u':8,'v':8, 'w':9,'x':9,'y':9,'z':9,
}

def build_index(words=None):
    """Build T9 digit-sequence → word lookup from the shape."""
    if words is None:
        words = unfold()
    index = {}
    for w in words:
        key = tuple(KEYPAD[c] for c in w)
        index.setdefault(key, []).append(w)
    return index

def solve(digits, index=None):
    """Given T9 digits (e.g. [4,3,5,5,6]), return matching words."""
    if index is None:
        index = build_index()
    return index.get(tuple(digits), [])


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    words = unfold()

    if "--tree" in sys.argv:
        print(f"\n  THE SHAPE ({len(S)} chars → {len(words)} words)\n")
        for block in S.split("^"):
            head, _, tail = block.partition("]")
            parts = tail.split("|")
            print(f"  {head}─┬─ {parts[0]}")
            for p in parts[1:-1]:
                print(f"  {' ' * len(head)} ├─ {p}")
            print(f"  {' ' * len(head)} └─ {parts[-1]}")
        print()

    elif "--solve" in sys.argv:
        index = build_index(words)
        print(f"\n  T9 Solver ({len(words)} words, {len(index)} unique key sequences)\n")
        # Demo
        demos = [
            [8,4,3],      # the/tie
            [4,6,6,3],    # good/gone/home
            [2,2,5,5],    # ball/call
            [5,6,6,9],    # know/joy?
            [5,8,7,8],    # just
            [8,4,3,6],    # them/then
        ]
        for d in demos:
            matches = solve(d, index)
            print(f"  {''.join(map(str,d))} → {', '.join(matches)}")
        print()

    else:
        print(f"\n  shape = \"{S[:50]}...\"")
        print(f"  len(shape) = {len(S)}")
        print(f"  words = unfold(shape)")
        print(f"  len(words) = {len(words)}\n")
        # Show in columns
        for i in range(0, len(words), 12):
            row = words[i:i+12]
            print("  " + " ".join(f"{w:<6s}" for w in row))
        print()
