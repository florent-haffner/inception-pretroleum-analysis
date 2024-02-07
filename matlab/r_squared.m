function [r2]=r_squared(y_true, y_preds)
%***************************************
% R-squared (Coefficient of determination)
%***************************************
% y_true = the observed values
% y_preds = the predicted values to measure

% Return a score between [-infinite, 1]
% A score close to 1 means the predictions are able to extract rules from data
% A score around 0 or negative means the predictions aren't able to extract rules

sum_square_residuals = sum((y_true - y_preds).^2);
total_sum_square = sum((y_true - mean(y_true)).^2);

r2 = 1 - (sum_square_residuals/total_sum_square);
