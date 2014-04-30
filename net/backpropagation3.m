function [W epochs] = backpropagation3(W,train,lab,f,df,error,beta=0.1,alfa=0.9)
	epochs=0;
	count=0;
	betamax=beta;
	Wl=W;
	klasy=getlabels(lab);
	min_klasa=min(klasy)
	do
		epochs++;
		E=0;
		for (t=1:size(train,2))
			x=train(:,t);
			[y dy]=feedforward2(W,x,f,df);

			%l=zeros(max(klasy)+1,1);
			%l(lab(t)+1)=1;

			%l=zeros(size(W{end},2),1);
			%l(lab(t)-min_klasa+1)=1;

            l=lab(t);

			E+=sum(l-y{end}).^2;
			
			d{size(W,2)}=(l-y{end}).*dy{end};

			for (i=size(W,2)-1:-1:1)
				d{i}=(W{i+1}*d{i+1})(1:end-1,:).*dy{i};
			end

			Wt=W;
			W{1}+=beta*[x;1]*d{1}'+alfa*(W{1}-Wl{1});
			for (i=2:size(W,2))
				W{i}+=beta*[y{i-1};1]*d{i}'+alfa*(W{i}-Wl{i});
			end
			Wl=Wt;
			
			%W{1}+=[x;1]*d{1}';
			%for (i=2:size(W,2))
			%	W{i}+=[y{i-1};1]*d{i}';
			%end
		end
		E
		fflush(stdout);	
		save net.tmp W
	until (E<error)
    epochs
