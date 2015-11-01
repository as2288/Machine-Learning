function [J, grad] = costFunction(theta, X, y)
%theta are the initial parameters of the system
%X are all the feature vectors
%y are the corresponding outputs

% Initialize some useful values
m = length(y); % number of training examples


J = 0;
[m,n] = size(X);
J = sum((log(sigmoid(X*theta)) .* -1 .* y) - (1.-y).*(log(1 - sigmoid(X*theta))))/m;
grad = zeros(size(theta));

num_iters = 1;
%running the gradient descent algorithm only once
alpha = 0.01;

h = sigmoid(X*theta);

%taking a gradient descent step
grad = (1/m)*(X' * (h-y));



end


