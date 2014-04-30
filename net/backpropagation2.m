function [W epochs] = backpropagation2(W,train,lab,f,df,Emax,beta=0.1,alfa=0.9)
	epochs=0;
	count=0;
	betamax=beta;
	Wl=W;
	do
		epochs++;
		E=0;
		for (t=1:size(train,2))
			x=train(:,t);
			[y dy]=feedforward2(x,W,f,df);
			E+=sum(lab(t)-y{end}).^2;
			
			
			d{size(W,2)}=(lab(t)-y{end}).*dy{end};

			for (i=size(W,2)-1:-1:1)
				d{i}=(W{i+1}*d{i+1})(1:end-1,:).*dy{i};
			end
			Wt=W;
			W{1}+=beta*[x;1]*d{1}'+alfa*(W{1}-Wl{1});
			for (i=2:size(W,2))
				W{i}+=beta*[y{i-1};1]*d{i}'+alfa*(W{i}-Wl{i});
			end
			Wl=Wt;
		end
		E
		fflush(stdout);	
	until (E<Emax || epochs>20000)
