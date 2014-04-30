cd set;
[tvec tlab tstv tstl]=readSets('a');
[mu trmx]=prepTransform(tvec,80);
tvec=pcaTransform(tvec,mu,trmx);
tstv=pcaTransform(tstv,mu,trmx);
save mu.txt mu
save trmx.txt trmx
