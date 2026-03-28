from zlib import decompress
from base64 import b85decode
S=decompress(b85decode(
    b"c-l=>iH@W?4E)kqq9GiQkVH9#Vdam1RSvycsb|y;@ll7J$$f&?x#_a#=5<SY6b(T`O?qk6xM+N|_({u!556|7r#5X~@9B}x<mOWm%OD;;wP^L)cKLXxQ_%^z)knUsO+JUSqh(S22rk{)y?m16u}80uMz0A;B2wtDqC!Umepamy>f*uUWS$(Gy!7&VW%QX*7CDP&@rw3j_BbN*?v5z);o2i}aQ!3msO2>*j7F^#G(Q%KsN(#sDa+hlMwVrA$O^y{5kIncH!V5^BJ)|61Fu1wEXjd`RP>Ir6gM%4_KgwUv7bi}Iq${@&`fy7Hpn>$LDL}b2Q>ZoqCZE_$IA+CO&k0iszjo^3?JH&6)DjMItYAw%X&DdIhdgso~RhGBjCsDudIRS!TKX>a%s`jM$TSj1$Gc}$=cn>Nd0A-Kq0BBNFkedO}4|Whzs@r8F2{s!3tWbBsQ$3BSqf8#z&_|$?Gj!cYKNZJE<kE_3NF@Bt*NzUr4^wL(6NUU?Kxhk{!H*xt)=AUiRcz7w8}MnuiMj3$*O8khk<r4syD{&+AKQif_G3Tb2JH5*+c-1k4@Q>gCZ6W1QYW9iKJRgXb}9|7~2(NbUy2FlOS>W6{&ULew!bhz({T@V9AY=izW(SI;Lr?Kwl4=Yc;Yrb6c&iN(BK2Qa9KN*%A|B9WkjHi^su7HlF;o_^%=rp<)*IT)9j<n5}$6IUXc7v%yOHV3SIA^8rB0DocR;^Zzb(JY6a;JOAlHxe|7#%jcC*nqCH6<~lRb{GZ~(Be|T`ns}FXa#f411}_tEkQH-8;bR2V}Mc0V+2m_Ca?*3G>oUefsVUq(6Iw%Mr)^yNRw>(yi6FSlHt=xAHwmyN=^=ngAhoF1%S!TmT#CwGF*$4ooj}SV1T4pGw31On)3r+`DY_6bRYOC#)q-DnDHpwDJAhJ8XH5JRty{4$+oqSs+%&XCY745{<RR#H`;nk$xt7VD{6BxxB>w8vOW8EoG5nPVXh-{8zr0jj%UNXn4idOwz5sEU;X2yo*Z}N4vulR)advh%C7B|4*gul>}HMuV?k<WTtfOri*VpsR;t<PXjBH-Z+BeMuu{So*lGQq#=<ELkgo6Ud6yj=)^<}5P+yq-F*j6d**HPm1fG{wW}Sw1ZN+m9eV4rE%A2H{wVG`^K<wP>t-FF}w_%X>xa_67h_(@BD}oYwj$n}=>jjN@EQf=h;NW8+-xC+jVT1hQde`f?z;+7v$jyN~?F#vO=)JY>o%<R161{<=w&+%a@+BpXT{f<|omcIhT;%wU8<3O1igLrmSsus%e@r|63imTipE^E8{<swV|6A2|?E"
)).decode()
unfold=lambda s=S:[p+x for b in s.split("^")for p,_,t in[b.partition("]")]for x in t.split("|")]
def t9(W=None):
 K="22233344455566677778889999";d={}
 for w in(W or unfold()):d.setdefault("".join(K[ord(c)-97]for c in w),[]).append(w)
 return d
if __name__=="__main__":
 import sys;W=unfold();a=sys.argv[1:]
 if a:ix=t9(W);[print(q,"->",", ".join(ix.get(q,["?"])))for q in a]
 else:print(f"{len(W)} words\n");print(*W)
