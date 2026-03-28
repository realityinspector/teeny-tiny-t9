from bz2 import decompress
from base64 import b85decode
S=decompress(b85decode(
    b"LRx4!F+o`-Q(3CdxIq8_YJ-3P13&-Y^iTwT-1l#9Ugjv-uRTCC(4z@6(HNA_8lIX23PgG}Jwpvs@QO475E=xj`ic<K%6c^b&?MA|(KRzBiM0{GPg-~0j#<|C&mS`|XP#yD@{08u69P3%`;ieUY>JjG?b*Mcx2qO4Y)g;#E5u@ry2TO&Z^(>zcZuq9TKlFn9s7QNe=ky&cRG;!HMCCO%$ri}BWsIU<B2TN@gUJb;&)x?HJf1c_c>_SO?1*Yn&Q>zi`yJ_!B5GwLI@~=Dne4CiH4(NA3x!Ce)#*baQscJpKF$3Nd>%^F{cVB9El4qN)QJht-=gdJs}<=5@<>QYLpwg(C&6;tI~PveB<9Soa0*e=jsi+|G7BM{|{MF<?15GHiI3iWYEI@l{r4rx{dmOv<Om+OhiwhO_N6h^$hEpN9R=7nj|$lH1Qp1-J+XXnzdcI#|cS7dajNE2Af20ViV+BGV*c~MNy5jq@<)e_4jN{8X#o)Yp@#&Z%4<K`xmbte9U%<;Tce<Nf|0g0{!<UP|gsHY=cE1O0l4rCn{7CNW?X#U!C{dm7Q6FiyKixAS@{b`g{ja;MPFmGluqb!>vg}wHVTCshO>0OF1{|#@$Sg+7d1hg?&_%f|uHMT_QIpDMlOFCP@lbK$kA;*0)?^&tnm;{nu7c2t*`s%77LVOi`tJ^!wn`0Z~zS&)yrBLzfrJJ-qS1k3Rk$9r)f9!YL;0kh3F}S0T7>nG59FtFmCo;R2W=8BIY;CW0EWTyfpBn~0@YyGR$BVqkd6No#!Os>%c)5rU;8vb+RS+a63k-MDhxZd;_JooaxR5!7tMrU6Z`-z%M0Q||iJu?Uwa0g|9dlDQ-adSemi+b&r^@y>Y0MIw@tCjVNzkI0I@Gvd7^LEme$zo&uJxn&Tr#$+AVY{aRl!xt#iksEx6RqqYT>aRte@rq~RN2e-U0akc`kcO2wthJtFoVuDK_q>tYm0*RNc{6K&MzL@3_j|ow9o->)%)~@e>4l0x0zwE{LtfzM<y;kc6Pa&!wFt;VQpKNKL5aPpXO1^`&q1y?KKKo-C*1efi4wz?wbzVs&THh>*Hp#TX6;mF^L^FI^E;y;z}$JAKI3ji*r`lgR@R$9!5Wqp4IATUZHl`~Z0lHJRZO&nzbrODqmHxb^hJoPe0KHEj)~_PWt{S=7mmo)up6l}fN^lKVWoRb*~_(4JdW!>g6>GC3KA7r_Xr>"
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
