import bz2,base64
S=bz2.decompress(base64.b85decode(
    b"LRx4!F+o`-Q&}Yvv}FJRG#P*Z1b6^H|KIRX1LSCy?Yeu;YBV)H5LEmHJVe??kZPZh5GtRFfsg}CMAXSl$*G_KN}r_&X_W!B4GjW9N_i%pQhG<IG!F*M=8LoUO<!j|<Js*I#Fz(>6Ei0VgNi*`&A|a8Y1+V#W^KGhJe;F>ZL)o_H`S_&l<5%G^v^KEobib6Qf1NB7QN4yT^oEbe=fyLzPBKeGgCVc$t?tlw!T9f21?6fo7Z4)lB<5C+oR%gDC;|VCIf8&oTEz1Pk8q7u`L!G>$g)=jQG7?&zA^Ypur&KxaM+3*9am(+4f-_G4vHYcRY7J=Mb@OnoETP$DZ|6pEmW1y~@P;rp*nUIADZYa|ckdol%&#P;BI-w*y;Nfy=*Jn1`HJ)0)#!=7JkBl@{E7hGm_|5`wBy?W+o_Y*(jO-rfd{P9h9+q|t<Ek+}hFFOxCQ@EHY;0!Jeh6UTc7%_Rcc`8He;<PTrl%XY&~m1UP>4ds2Z=M>*?0|OWV5HRG&4f{L|P1(JaZ%KYptoWMpRm+4U-OcykPkZ_I`Cl4piLj{!skiTpb*??!V1mGdj6K{J^`ceL2AnVw6o;#)!`}4NG!q0KyUlpxH%@;iTE{IGZNQk;T!ZB*{q)zu9wB5=x#~JNTddTsVQy5fJTj6AE{R@xZ8C}V;dXiym6}->tl<nnYWVt=x*!TML$bDVb(9mr0^}q_gchkE)N=<MJwn-3il|jpeciWAD90m-9J>@)+kJtwwoNBZL(k1w*FN-mygnH^>slmAN8?hwCuQiFYMWanInMn(rs=F+f~!SQ3?#ka^Na^Hs(v|kMIv3yg$9tyL=j5oB8{aRkqo^WF&Xizwl$$zVxtl~m4%oT&Am1q5f~Eb9rJ+^vOsQ8i@rIQLmC>sD5N*p$L2(s&kw`Tex`9<$BQxac&(Z3_TxZ|j98s-M1e*z7$+qNsW;WBWGD?C8jWnr*s*HkqK-!-n9NAKKw*ca>bYfqLCaCSK+}&-XGdr`_*NJ?hl0~sjqeD?>#4mEAJj)j#&1Aoab;l@LhKY#7|<bTCo++qTuVz4y}|)`%|<M>O<AlTZL`x@&4<3`)BIh@6yZWZluFT*"
)).decode()
def unfold(s=S):
 W=[];p=''
 for d in s.split(','):w=p[:int(d[0])]+d[1:];W+=[w];p=w
 return W
def t9(W=None):
 K="22233344455566677778889999";d={}
 for w in(W or unfold()):d.setdefault("".join(K[ord(c)-97]for c in w),[]).append(w)
 return d
if __name__=="__main__":
 import sys;W=unfold();a=sys.argv[1:]
 if a:ix=t9(W);[print(q,"->",", ".join(ix.get(q,"?")))for q in a]
 else:print(f"{len(W)} words\n");print(*W)
