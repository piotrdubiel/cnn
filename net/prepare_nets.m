cd set
[tvec tlab tstv tstl]=readSets('a');
[mu trmx]=prepTransform(tvec,150);
tvec=pcaTransform(tvec,mu,trmx);
tstv=pcaTransform(tstv,mu,trmx);

tlab(tlab==38)=-1;
tlab(tlab==39)=38;
tlab(tlab==-1)=39;

tstl(tstl==38)=-1;
tstl(tstl==39)=38;
tstl(tstl==-1)=39;


tlab(tlab==73)=-1;
tlab(tlab==74)=73;
tlab(tlab==-1)=74;

tstl(tstl==73)=-1;
tstl(tstl==74)=73;
tstl(tstl==-1)=74;

cd ../nets-v1
save mu.txt mu
save trmx.txt trmx
cd ..

fprintf(1,'Training digits net\n')

dv = tvec(tlab < 10, :)';
dl = tlab(tlab < 10)';
dtv = tstv(tstl < 10, :)';
dtl = tstl(tstl < 10)';

Q1={rand(151,80),rand(81,10)};
digit_net = train(Q1,dv,dl,dtv,dtl,30);
cd nets-v1
save digit_net.txt digit_net
cd ..

clear dv dl dtv dtl

fprintf(1,'Digit net DONE\n')



fprintf(1,'Training small letter net\n')

sv = tvec(tlab >= 45, :)';
sl = tlab(tlab >= 45)';
stv = tstv(tstl >= 45, :)';
stl = tstl(tstl >= 45)';

Q1={rand(151,120),rand(121,35)};
pl_small_net = train(Q1,sv,sl,stv,stl,70);
cd nets-v1
save pl_small_net.txt pl_small_net
cd ..

clear sv sl stv stl

fprintf(1,'Small letter net DONE\n')



fprintf(1,'Training capital letter net\n')

cv = tvec(tlab >= 10 & tlab <= 44, :)';
cl = tlab(tlab >= 10 & tlab <= 44)';
ctv = tstv(tstl >= 10 & tstl <= 44, :)';
ctl = tstl(tstl >= 10 & tstl <= 44)';

Q1={rand(151,100),rand(101,35)};
pl_capital_net = train(Q1,cv,cl,ctv,ctl,50);
cd nets-v1
save pl_capital_net.txt pl_capital_net
cd ..

clear cv cl ctv ctl

fprintf(1,'Capital letter net DONE\n')


fprintf(1,'Training letter or digit net\n')

lodv = tvec';
lodl = (tlab >= 10)';
lodtv = tstv';
lodtl = (tstl >= 10)';

Q1={rand(151,100),rand(101,2)};
lod_net = train(Q1,lodv,lodl,lodtv,lodtl,50);
cd nets-v1
save lod_net.txt lod_net
cd ..

clear lodv lodl lodtv lodtl

fprintf(1,'Letter or digit net DONE\n')



fprintf(1,'Training capital or small net\n')

cosv = tvec(tlab >= 10, :)';
cosl = tlab(tlab >= 10)';
costv = tstv(tstl >= 10, :)';
costl = tstl(tstl >= 10)';

cosl = cosl >= 45;
costl = costl >= 45;

Q1={rand(151,100),rand(101,2)};
cos_net = train(Q1,cosv, cosl, costv, costl,50);
cd nets-v1
save cos_net.txt cos_net
cd ..

clear cosv cosl costv costl

fprintf(1,'Capital or small net DONE\n')

