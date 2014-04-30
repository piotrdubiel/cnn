function Q2 = train(Q1,set,lab,tstv,tstl,error)
	Q2=backpropagation3(Q1,set,lab,@sigm,@dsigm,error,0.9,0.6);
	'Teaching done'
	fflush(stdout);
	save Q2.txt Q2
	y=feedforward2(Q2,tstv,@sigm,@dsigm); 
    y
	[i ix]=max(y{end});
	ix=ix+min(getlabels(lab))-1; 
	quality=sum(ix==tstl)/size(tstv,2)
