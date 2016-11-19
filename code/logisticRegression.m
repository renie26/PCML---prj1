function [beta] = logisticRegression(y, tX, alpha)

% algorithm parametes
maxIters = 2000;


dimension  = size(tX,2);
beta = rand(dimension, 1)*0.001;
%beta = [0.7; -1.7; 0.038];
L_all = [];
% iterate
fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 beta1 beta2\n');
for k = 1:maxIters
    %compute cost
    mle = y'*tX*beta - sum(log(1+exp(tX*beta)));
    L = -mle;
    
    %compute gradient
    temp = tX*beta;
    s = exp(temp)./(1+(exp(temp)));
    e = y-s; %compute error
    g = tX'*e;
    
    %compute Hessian
    S = zeros(length(s));
    for i=1:length(s)
        S(i,i)= s(i)*(s(i)-1)';
    end
    %disp(size(S))
    H = tX'*S*tX;
    
	%using Newton's method to compute beta
    beta = beta - alpha *  (H\g);
    
    % INSERT CODE FOR CONVERGENCE
    if(k>2 && (L>(1 - 1e-10)*L_all(k-1)))
        disp('Already converged')
        break;
    end
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;
end