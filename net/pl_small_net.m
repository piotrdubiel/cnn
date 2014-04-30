cd set;
[tvec tlab tstv tstl]=readSets('a');
[mu trmx]=prepTransform(tvec,120);
tvec=pcaTransform(tvec,mu,trmx);
tstv=pcaTransform(tstv,mu,trmx);
save mu.txt mu
save trmx.txt trmx

tvec = tvec(tlab >= 45, :);
tlab = tlab(tlab >= 45);
tstv = tstv(tstl >= 45, :);
tstl = tstl(tstl >= 45);

tvec=tvec';
tlab=tlab';
tstv=tstv';
tstl=tstl';

cd ..
Q1={rand(121,90),rand(91,35)};
train
pl_small = Q2;
save pl_small.txt pl_small

