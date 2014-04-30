function [y z dy dz] = feedforward(x,W,V,f,df,l=1)
	x=[x;repmat(1,1,size(x,2))];
	net=W'*x;
	z=f(net,l);
	dz=df(net,l);
	net=V'*[z;repmat(1,1,size(z,2))];
	y=f(net,l);
	dy=df(net,l);
