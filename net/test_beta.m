function [m epoki]= test_beta(train,lab,Emax,betastart=1,control=1)
	for (i=1:10)
		w=(rand(3,2)-0.5)/sqrt(2)
		v=(rand(3,1)-0.5)/sqrt(2)
		[W V e]=backpropagation(w,v,train,lab,@sigm_uni,@dsigm_uni,Emax,betastart,control);
		epoki(i)=e;
	end
	m=mean(epoki);

