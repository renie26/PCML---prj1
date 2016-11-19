function [beta] = penLogisticRegression(y, tX, alpha, lambda)

% algorithm parametes
maxIters = 2000;


dimension  = size(tX,2);
beta = rand(dimension, 1)*0.001;
%beta = [0.7; -1.7; 0.038];
% iterate
for k = 1:maxIters
    %compute cost
    mle = y'*tX*beta - sum(log(1+exp(tX*beta)));
    L = -mle;
    
    %compute gradient
    temp = tX*beta;
    s = exp(temp)./(1+(exp(temp)));
    e = y-s; %compute error
    g = tX'*e; %compute gradient
    
    %compute Hessian
    S = zeros(length(s));
    for i=1:length(s)
        S(i,i)= s(i)*(s(i)-1)';
    end
    H_1 = tX'*S*tX;
    
    H_2 = lambda*eye(length(beta)); % consider the regularization term
    H_2(1,1) = 0;
    H = H_1+H_2;
    
    % update beta using newton's method for penalized logistic regression
    beta = beta - alpha *  (H\g);
    
    % check convergence
    if(k>2 && (L>(1 - 1e-10)*L_all(k-1)))
        disp('Already converged')
        break;
    end
end

  if(k == maxIters)
        disp('Reached the maximum number of iterations \n')
  end