function [beta] = ridgeRegression(y,tX,lambda)

gram = tX'*tX+lambda*eye(size(tX,2));
%inv_gram = inv(gram);
beta = gram\(tX'*y);
