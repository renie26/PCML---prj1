function [beta] = leastSquares(y, tX)

gram = tX'*tX;
%inv_gram = inv(gram);
beta = gram\(tX'*y);