#!/usr/bin/env python3
"""10 lines of Python -> full T9. A 1KB binary seed unfolds to 617 English words."""
from zlib import decompress
from base64 import b64decode

S = decompress(b64decode(
    b"eNpNVYmOpDoM/NJYoiEcj5BEOYZh5Y9/VQ69Wqln1A3xUYedyX2C17npsujmdUt6FA1BQ016adS4aHy0+EnLhQ++Nq2nNm1e7+mRz+TmUxHLIB4+tWj12nb5eKdT0Qm51Ue+r00+h7OjZVH8CC7d2r18kuOxR6+PRq8JSSJS6K+iQkcEflatD+riweNknhybvPTyemX0mVEWOYtn8Yr2ZPY4IvPujijz4dojc3D9I/Oo5TUsjEatFDQ/Foqo4vytKcvcXUaWZXLICsATET+yeDctOgUi81mWA69BmyzJgYNU9I6yFDcxh/aNEd2xn0cQOe/GCNBMeMY2yDkIQdMg7wc0/fiiP0fQx8sKbk0b/ByqRKK8DB/akRUp0QY6CHx7y3o4qDmYoRieqBiwI+DH66+sgejB+iNrcmnR1EbOXmQFdhByydrdyEj9ZZtAIamqSCGbd022w62IC/YAGXEI+MEKWqkkNqymI0XewMZHp0fJ6y1bd3yJ+jupJa71653NJES7w19IvtNDy2ujYCIDUJIdvIO5nZF2XvbkWBfoH/aahwsAC7TssJC39E0OcrrAz+tLaIpvvUtblf8gEhzx0QTMHdbq5dEOY5zeQR2fNdOD5zFs/21czkjuBsQwZuLRY9HTv0LUMTKIDWYh8LdpZefhcF5Xzyc87TlDmeeBPySmQuccicW8XQcwqIlKHa/lIpGggzaJzAEWbCJOnn7k8qbNhPY5GJeZ5O2qwhNXIpkX5QIZNgesfSH5DowAf2Hk4uQuL9Esh57gugY4NxI0icZqxGvYgK6zFWCAJXYYBhXQ4lLB+rqyUWgFc80DLySLLUEz+huK9WZjgFHKYwRO7iawCXg0jKe/gY67aNCxA2ZGa6eNSJNs+rxMGKWS4dNIH8LxvUpODoqDEEiIBYDJzZZvmEYyXIuUnWuGjGCApFgz6CHamvMDX/lnIyx0H/gqxoext5jxKNwpJfEoGhtDmlYMG9vJtrWQqpsyqCaVmq7+tRDPN9p6TMlQH5AranMyuAIA97I9+3ecKHbdYa4jcxslm/uOZ8O8mJ/IbLayTaxf/eOlnhx35A6OcbfU6Pgfyq7tlQ79XMbsZo4BY0FqZlxC+kaLcOBZVqrZ6GhUBtlvd1zSsFLCF9hp/Z7GwikNkK7v7nmXHRzWdlMPUF+Y/jE3VGrZOEHcBJ6XVWadhnZtxXMNfZe85sqbpJu1mm07QOzc6627gme3YZD2uOylu4i2EWUrghJ6jmCRH+sFjXksh4Oe4OBjId8TF81hg/zietfruMB2CvfIzYE87Qr5e+i9te6d+xn4Er2Bo7boVoPyJSWOXTauup1X7Z1c5PiOG5CTwYpyw8RZHsgBf0w7/gr3M0yfOj5F/rgU/f9b1Xbt"
)).decode()

unfold = lambda s=S: [p+x for b in s.split("^") for p,_,t in [b.partition("]")] for x in t.split("|")]

def t9(W=None):
    K = "22233344455566677778889999"
    d = {}
    for w in (W or unfold()):
        d.setdefault("".join(K[ord(c)-97] for c in w), []).append(w)
    return d

if __name__ == "__main__":
    import sys; W = unfold()
    if sys.argv[1:]:
        ix = t9(W)
        for q in sys.argv[1:]: print(q, "->", ", ".join(ix.get(q, ["?"])))
    else: print(f"{len(W)} words\n"); print(*W)
