function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
len_theta = size(theta);
grad = zeros(len_theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% y' 1xm * log(h_x) mx1 -> 1x1
% - (1-y)' 1xm * (1-h_x) mx1 -> 1x1
% 1/m * 1x1 => number

% Find standard component of cost function
h_x = sigmoid(X * theta);
J_std = (1 / m) * (-y' * log(h_x) - (1 - y)' * log(1 - h_x));

% Find regularization component of cost function (NOT theta_0!)
theta_to_reg = theta(2:len_theta);
J_reg = (lambda / (2*m)) * sum(theta_to_reg .^2);

% Add standard and regularization components for full cost function
J = J_std + J_reg;

% X' 1xm * h_x-y mx1 => 1x1
grad(1) = (1/m) * X(:, 1)'*(h_x - y);
% X' nxm * h_x mx1 => (n+1)x1
grad(2:len_theta) = ((1/m) * (X(:, 2:len_theta)' * (h_x - y))) + (lambda / m) * theta_to_reg;




% =============================================================

end
