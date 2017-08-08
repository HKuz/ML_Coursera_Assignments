function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% mx(n+1) * (n+1)x1 -> mx1 matrix h(x), - mx1 vector y, element-wise square
% temp = ((X * theta) - y).^2;
% J = (1/(2*m)) * sum(temp);

% Vectorized version
J = (1/(2*m)) * ((X * theta - y)' * (X * theta - y));


% =========================================================================

end
