function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 is 25 x 401
% Theta2 is 10 x 26

% Add ones to the X data matrix, so it's 5000 x 401
a1 = [ones(m, 1) X];

% Find z(2) = a(1) (5000 x 401) * Theta1' (401 x 25) => z(2) (5000 x 25)
z2 = a1 * Theta1';

% Add ones to the z2 data matrix, take g(z2)
a2 = [ones(m, 1) sigmoid(z2)];

% Find z(3) = a(2) (5000 x 26) * Theta2' (26 x 10) => z(3) (5000 x 10)
z3 = a2 * Theta2';

% Find max value and get its index for digit prediction
[maxes, p] = max(z3, [], 2);


% =========================================================================


end
