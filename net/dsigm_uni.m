function y = dsigm_uni(x,l)	
	y=sigm_uni(x,l).*(1-sigm_uni(x,l));
