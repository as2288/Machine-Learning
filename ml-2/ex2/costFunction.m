function [J, grad] = costFunction(theta, X, y)
	%theta are the initial parameters of the system
	%X are all the feature vectors
	%y are the corresponding outputs

	% Initialize some useful values
	m = length(y); % number of training examples
	J = 0;
	[m,n] = size(X);
	h = sigmoid(X*theta);

	%J is the cost of using theta
	J = sum((log(h) .* -1 .* y) - ((1 .- y).*(log(1 - h))))/m;

	grad = zeros(size(theta));
	num_iters = 1;
	%running the gradient descent algorithm only once
	
	h = sigmoid(X*theta);
	%taking a gradient descent step
	grad = (1/m)*(X' * (h-y));
end


