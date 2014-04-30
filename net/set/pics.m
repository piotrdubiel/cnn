function pics(set)
	%for (i=1:size(set,1))
		X=zeros(28,28);
		for (p=1:28)
		for (q=1:28)
			X(p,q)=set(1,28*(p-1)+q);
		end
		end
	    %	imshow(X);
		file=strcat(num2str(1),".png");
		print(file,'-dpng');
	%end
