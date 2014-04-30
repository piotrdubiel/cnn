function [W V epochs] = backpropagation(W,V,train,lab,f,df,Emax,beta=1,control_beta=1)
	epochs=0;
	El=0;
	count=0;
	betamax=beta;
	do
		epochs++;
		E=0;
		for (t=1:size(train,2))
			x=train(:,t);
			[y z dy dz]=feedforward(x,W,V,f,df);
			E+=sum(lab(t)-y).^2;
			d{2}=(lab(t)-y).*dy;
			d{1}=(V'*d{2})(:,1:end-1).*dz';

			V+=beta*d{2}*[z;1];
			W+=beta*[x;1]*d{1};
		end
		if (control_beta==1)			
			if (E>El) 
				beta*=0.1;
				count=0;
			elseif (E<El)
				beta+=0.1*count;
			end
			beta
			if (beta>betamax)
				betamax=beta;
			end
		end
		E
		El=E;

		fflush(stdout);	
	until (E<Emax || epochs>20000)
