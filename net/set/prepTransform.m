function [mu trmx] = prepTransform(tvec, comp_count)
% tvec - maciery zawieraj¹ca obrazy zbioru uczacego
% comp_count - liczba najwiêkszych sk³adowych w macierzy transformacji
% mu - warto¶æ ¶rednia zbioru ucz±cego
% trmx - macierz transformacji do comp_count-wymiarowej przestrzeni PCA
% Funkcja zapisuje w plikach wektor sredniej zbioru ucz¹cego, oraz WSZYSTKIE wektory w³asne 
% w kolejno¶ci od odpowiadaj¹cego najwiêkszej wartosci w³asnej (w tej wersji zapis wykomentowany!)

mu = mean(tvec);
cmx = cov(tvec);
[evec eval] = eig(cmx);
eval = sum(eval);
[eval evid] = sort(eval);
eval = eval(size(eval,2):-1:1);
evec = evec(:, evid(size(eval,2):-1:1));
% save 'nist_mu.mat' mu
% save 'nist_cov.mat' evec 
trmx = evec(:, 1:comp_count);
