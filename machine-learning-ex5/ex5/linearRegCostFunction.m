function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_x = X * theta;

% Find cost function
% Find standard component of cost function
J_std = (1/(2*m)) * ((h_x - y)' * (h_x - y));

% Find regularization component of cost function (NOT theta_0!)
J_reg = (lambda / (2*m)) * sum(theta(2:end) .^2);

% Add standard and regularization components for full cost function
J = J_std + J_reg;

% Find regularized linear regression gradient
% X' 1xm * h_x-y mx1 => 1x1
grad(1) = (1/m) * X(:, 1)'*(h_x - y);
% X' nxm * h_x mx1 => (n-1)x1
grad(2:end) = ((1/m) * (X(:, 2:end)' * (h_x - y))) + (lambda / m) * theta(2:end);


% =========================================================================

grad = grad(:);

end
