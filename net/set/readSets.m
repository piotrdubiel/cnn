function [tvec tlab tstv tstl] = readSets(dla='a')
% funkcja czyta zbiory ucz¹cy i testowy
% powsta³a, bo zmêczy³o mnie wpisywanie potwornych nazw plików

	fnames = { 'train-images.idx3-ubytes'; 'train-labels.idx1-ubytes';  'test-images.idx3-ubytes'; 'test-labels.idx1-ubytes' };
	%fnames = { 'train-images0-44.idx3-ubytes'; 'train-labels0-44.idx1-ubytes'; 'test-images0-44.idx3-ubytes'; 'test-labels0-44.idx1-ubytes' };

	[tlab tvec] = readmnist(fnames(1,1), fnames(2,1));
	[tstl tstv] = readmnist(fnames(3,1), fnames(4,1));

	if dla == 'd' || dla == 'digits'
		tvec = tvec(tlab < 10, :);
		tlab = tlab(tlab < 10);
		tstv = tstv(tstl < 10, :);
		tstl = tstl(tstl < 10);
	elseif dla == 'b' || dla == 'big'
		tvec = tvec(tlab <= 44, :);
		tlab = tlab(tlab <= 44);
		tstv = tstv(tstl <= 44, :);
		tstl = tstl(tstl <= 44);
	elseif dla == 'l' || dla == 'letters'		
		tvec = tvec(tlab >= 10 & tlab <= 44, :);
		tlab = tlab(tlab >= 10 & tlab <= 44);
		tstv = tstv(tstl >= 10 & tstl <= 44, :);
		tstl = tstl(tstl >= 10 & tstl <= 44);
	elseif dla == 's' || dla == 'small'
		tvec = tvec(tlab >= 45, :);
		tlab = tlab(tlab >= 45);
		tstv = tstv(tstl >= 45, :);
		tstl = tstl(tstl >= 45);
	else return;
	end
