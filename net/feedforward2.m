function [y dy] = feedforward2(W,x,f,df,l=1)
	x=[x;repmat(1,1,size(x,2))];
	for (i=1:size(W,2))
		net=W{i}'*x;
		y{i}=f(net,l);
		dy{i}=df(net,l);
		x=[y{i};repmat(1,1,size(y{i},2))];
	end
