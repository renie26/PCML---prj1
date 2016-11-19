function [beta] = leastSquaresGD(y, tX, alpha)

  % algorithm parametes
  maxIters = 10000;

  % initialize
  D  = size(tX,2);
  beta = rand(D, 1);
    
  L_all = [];
  % iterate
  for k = 1:maxIters
    e = y - tX*beta; %compute error
    g = -1/length(y)*tX'*e; % compute gradient

    L = e'*e/(2*length(y));%compute MSE

    beta = beta - alpha* g;

    % check convergency
    if(k>2 && (L>(1 - 1e-10)  *L_all(k-1)))
        disp('Already converged')
        break;
    end

  end
  if(k == maxIters)
        disp('Reached the maximum number of iterations \n')
  end
  

