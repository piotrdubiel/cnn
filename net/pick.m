letv=tvec(:,tlab == 10 | tlab==12 | tlab==13 | tlab==15 | tlab==16 | tlab==18 | tlab==19 | tlab==20 | tlab==21 | tlab==22 | tlab==23 | tlab==24 | tlab==26 | tlab==27 | tlab==29 | tlab==31 | tlab==32 | tlab==33 | tlab==34 | tlab==36 | tlab==37 | tlab==38 | tlab==39 | tlab==40 | tlab==41 | tlab==42);

letl=tlab(tlab == 10 | tlab==12 | tlab==13 | tlab==15 | tlab==16 | tlab==18 | tlab==19 | tlab==20 | tlab==21 | tlab==22 | tlab==23 | tlab==24 | tlab==26 | tlab==27 | tlab==29 | tlab==31 | tlab==32 | tlab==33 | tlab==34 | tlab==36 | tlab==37 | tlab==38 | tlab==39 | tlab==40 | tlab==41 | tlab==42);

ltv=tstv(:,tstl == 10 | tstl==12 | tstl==13 | tstl==15 | tstl==16 | tstl==18 | tstl==19 | tstl==20 | tstl==21 | tstl==22 | tstl==23 | tstl==24 | tstl==26 | tstl==27 | tstl==29 | tstl==31 | tstl==32 | tstl==33 | tstl==34 | tstl==36 | tstl==37 | tstl==38 | tstl==39 | tstl==40 | tstl==41 | tstl==42);

ltl=tstl(tstl == 10 | tstl==12 | tstl==13 | tstl==15 | tstl==16 | tstl==18 | tstl==19 | tstl==20 | tstl==21 | tstl==22 | tstl==23 | tstl==24 | tstl==26 | tstl==27 | tstl==29 | tstl==31 | tstl==32 | tstl==33 | tstl==34 | tstl==36 | tstl==37 | tstl==38 | tstl==39 | tstl==40 | tstl==41 | tstl==42);

numv=tvec(:,tlab<10);
numl=tlab(tlab<10);
ntv=tstv(:,tstl<10);
ntl=tstl(tstl<10);


letl(letl==10)=0;
letl(letl==12)=1;
letl(letl==13)=2;
letl(letl==15)=3;
letl(letl==16)=4;
letl(letl==18)=5;
letl(letl==19)=6;
letl(letl==20)=7;
letl(letl==21)=8;
letl(letl==22)=9;
letl(letl==23)=10;
letl(letl==24)=11;
letl(letl==26)=12;
letl(letl==27)=13;
letl(letl==29)=14;
letl(letl==31)=15;
letl(letl==32)=16;
letl(letl==33)=17;
letl(letl==34)=18;
letl(letl==36)=19;
letl(letl==37)=20;
letl(letl==38)=22;
letl(letl==39)=21;
letl(letl==40)=23;
letl(letl==41)=24;
letl(letl==42)=25;

ltl(ltl==10)=0;
ltl(ltl==12)=1;
ltl(ltl==13)=2;
ltl(ltl==15)=3;
ltl(ltl==16)=4;
ltl(ltl==18)=5;
ltl(ltl==19)=6;
ltl(ltl==20)=7;
ltl(ltl==21)=8;
ltl(ltl==22)=9;
ltl(ltl==23)=10;
ltl(ltl==24)=11;
ltl(ltl==26)=12;
ltl(ltl==27)=13;
ltl(ltl==29)=14;
ltl(ltl==31)=15;
ltl(ltl==32)=16;
ltl(ltl==33)=17;
ltl(ltl==34)=18;
ltl(ltl==36)=19;
ltl(ltl==37)=20;
ltl(ltl==38)=22;
ltl(ltl==39)=21;
ltl(ltl==40)=23;
ltl(ltl==41)=24;
ltl(ltl==42)=25;




