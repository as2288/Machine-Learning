function [all_theta] = oneVsAll(X, y, num_labels, lambda)


% Some useful variables
m = size(X, 1);
n = size(X, 2);

% initializing each parameter vector to 0 values
%     n        1 2 3 4 5 ..           400
% theta(for 0) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 1) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 2) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 3) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 4) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 5) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 6) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 7) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 8) 0 0 0 0 0 0 0 0 0 0 0 0 ..
% theta(for 9) 0 0 0 0 0 0 0 0 0 0 0 0 ..

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

for c=1:num_labels
	%  Set options for fminunc
	% use the gradient to minimize cost function
	options = optimset('GradObj', 'on', 'MaxIter', 50);
	initial_theta = zeros(n+1,1);
	% (y==c) treats c as the positive class and everything else as negative
	% in first iteration 1 is positive class etc..
	[theta, cost] = ...
	fminunc(@(t)(lrCostFunction(t, X, (y==c), lambda)), initial_theta, options);
	% we now have the parameter vector for theta for c
	% we can use this theta for predictions
	all_theta(c,:) = theta;
end


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%












% =========================================================================


end
