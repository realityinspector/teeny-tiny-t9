#!/usr/bin/env python3
"""2KB shape that unfolds to 617 English words, then solves T9."""

S = (
    "a]ble|ct|dd|ge|go|ir|ll|lso|m|n|nd|ny|rea|rm|rmy|rt|sk|t|te|way^ba]ck|d|"
    "g|ll|nd|nk|r|se|th^be]|ar|at|d|en|ll|st^bi]g|ll|rd|t^bl]ow|ue^bo]at|dy|m"
    "b|ne|ok|rn|th|x|y^bu]g|rn|s|sy|t|y^by]^ca]ll|lm|me|mp|n|p|r|rd|re|se|st|"
    "t^ce]ll^ch]in^ci]ty^cl]ub^co]at|de|ld|me|ok|ol|py|re|st^cr]ew|op^cu]p|t^"
    "da]d|rk|ta|te|y^de]ad|al|ar|ep^di]d|e|g^do]|g|or|wn^dr]aw|op|ug|y^du]st|"
    "ty^e]ach|ar|arn|ast|at|dge|gg|lse|nd|ven|ver|vil|ye^fa]ce|ct|il|ir|ll|n|"
    "r|rm|st|t|te^fe]ar|ed|el|ll|w^fi]le|ll|lm|nd|ne|re|rm|sh|t|ve|x^fl]at|ow"
    "|y^fo]od|ot|r|rm|ur^fr]ee|om^fu]el|ll|n|nd^ga]in|me|s|ve^ge]t^gi]ft|rl|v"
    "e^gl]ad^go]|al|d|es|ld|lf|ne|od|t^gr]ab|ay|ew|ow^gu]lf|n|y^ha]d|ir|lf|ll"
    "|nd|ng|rd|rm|s|t|te|ve^he]|ad|ar|at|lp|r|re|ro^hi]de|gh|ll|m|s|t^ho]ld|l"
    "e|ly|me|pe|st|t|ur|w^hu]ge|ng|rt^i]ce|dea|f|ll|n|ron|s|t|tem|ts^j]ack|ob"
    "|oin|ump|ury|ust^ke]en|ep|pt|y^ki]ck|d|ll|nd|ng^kn]ee|ew|ow^la]ck|dy|id|"
    "ke|nd|ne|st|te|w|y^le]ad|ft|g|ss|t^li]e|fe|ft|ke|ne|nk|p|st|ve^lo]ck|ng|"
    "ok|rd|se|ss|st|t|ve|w^lu]ck^ma]d|de|il|in|ke|le|n|p|rk|ss|y^me]|al|an|et"
    "|t^mi]le|nd|ne|ss|x^mo]de|m|od|on|re|st|ve^mu]ch|d|st^my]^na]me^ne]ar|ck"
    "|ed|t|w|ws|xt^ni]ce|ne^no]|ne|r|se|t|te|w^nu]t^o]dd|dds|f|ff|il|ld|n|nce"
    "|ne|nly|nto|pen|r|ur|ut|ver|wn^pa]ce|ck|ge|id|in|ir|le|lm|rk|rt|ss|st|th"
    "|y^pe]ak|n|r|t^pi]ck|e|le|n|ne|nk^pl]an|ay|ot|us^po]em|et|ll|ol|or|p|rt|"
    "st|t|ur^pr]ay^pu]ll|re|sh|t^ra]ce|in|n|nk|re|te|w^re]ad|al|ar|d|ly|st^ri"
    "]ce|ch|d|de|ng|se|sk^ro]ad|ck|le|ll|of|om|ot|pe|se|w^ru]le|n|sh^sa]d|fe|"
    "id|ke|le|lt|me|nd|ng|t|ve|w|y^se]a|at|ed|ek|em|en|lf|ll|nd|nt|t^sh]e|ip|"
    "op|ot|ow|ut^si]ck|de|gn|ng|nk|r|t|te|x|ze^sk]in|y^sl]ip|ow^sn]ow^so]|ft|"
    "il|ld|le|me|n|ng|on|rt|ul^sp]in|ot^st]ar|ay|ep|op^su]ch|it|n|re^sw]im^ta"
    "]il|ke|le|lk|ll|nk|pe|sk^te]am|ll|n|nd|rm|st|xt^th]an|at|e|em|en|ey|in|i"
    "s|us^ti]e|ll|me|ny|p|re^to]|e|ld|ne|o|ok|ol|p|ps|rn|ur|wn^tr]ee|ip|ue|y^"
    "tu]rn^tw]in|o^ty]pe^u]nit|p|pon|s|se|sed|ser^v]an|ast|ery|ice|iew|ote^wa"
    "]ge|it|ke|lk|ll|nt|r|rm|rn|s|sh|ve|y^we]|ak|ar|ek|ll|nt|re|st|t^wh]at|en"
    "|o|om|y^wi]de|fe|ld|ll|n|nd|ne|ng|re|se|sh|th^wo]n|od|rd|re|rk|rm|rn^wr]"
    "ap^y]ard|eah|ear|es|et|ou|our^z]one"
)

unfold = lambda s=S: [p+x for b in s.split("^") for p,_,t in [b.partition("]")] for x in t.split("|")]

def t9(W=None):
    K = "22233344455566677778889999"
    d = {}
    for w in (W or unfold()):
        d.setdefault("".join(K[ord(c)-97] for c in w), []).append(w)
    return d

if __name__ == "__main__":
    import sys
    W = unfold()
    if sys.argv[1:]:
        ix = t9(W)
        for q in sys.argv[1:]:
            print(q, "->", ", ".join(ix.get(q, ["?"])))
    else:
        print(f"{len(S)} chars -> {len(W)} words\n")
        print(*W)
