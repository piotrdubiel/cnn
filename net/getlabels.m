function labels = getlabels(labset)
	n=1;
	for (i=min(labset):max(labset))
		if (size(labset(labset==i))!=0)
			labels(n)=i;
			n++;
		end
	end
