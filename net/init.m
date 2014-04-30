cd set

[tvec tlab tstv tstl]=readSets('a');
'Loaded'
cd ..

fflush(stdout);

cd set

[mu trmx]=prepTransform(tvec,150);

'PCA Prepered'


save mu.txt mu

save trmx.txt trmx


#tvec=pcaTransform(tvec,mu,trmx);
#
#tstv=pcaTransform(tstv,mu,trmx);


tlab=tlab';
tstl=tstl';
tvec=tvec';
tstv=tstv';
cd ..

save tvec.txt tvec
save tlab.txt tlab
save tstv.txt tstv
save tstl.txt tstl
