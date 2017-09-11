function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
ranges = [0.01 0.03 0.1 0.3 1 3 10 30];

for i=1: length(ranges),
    C_temp = ranges(i);
    for j=1: length(ranges),
        sigma_temp = ranges(j);

        % Train model with temp values of C and sigma
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        current_error = mean(double(predictions ~= yval));

        % Set values for C and sigma if error is less than any other combo
        if i == 1 && j == 1,
            min_error = current_error;
        elseif current_error < min_error,
            min_error = current_error;
            C = C_temp;
            sigma = sigma_temp;
        end;
    end;
end;

% Optimal value for C = 1
% Optimal value for sigma = 0.1

% =========================================================================

end
