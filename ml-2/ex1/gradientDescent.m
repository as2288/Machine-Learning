function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    Delta1 = 0;
    Delta2 = 0;
    Delta1 = alpha*(1/m)*sum((X*theta - y) .* X(:,1))  ;
    Delta2 = alpha*(1/m)*sum((X*theta - y) .* X(:,2))  ;
    theta = theta - [Delta1; Delta2];
    computeCost(X, y, theta)
    J_history(iter) = computeCost(X, y, theta);

end

end
