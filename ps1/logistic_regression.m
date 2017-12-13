%logistic regression using Newton's Method

function[theta] = logistic_regression(X,Y)
m = size(X,1);
n = size(X,2);
theta = zeros(n,1)
last = ones(n,1)
while norm(last - theta) > 1e-5
	z = Y .* (X * theta);
	sigmoid = 1 ./ (1 + exp(z));
	gradient = -(1/m) * (X' * (sigmoid .* Y));
	hessian = (1/m) * (X' * diag(sigmoid .* (1-sigmoid)) * X);
	last = theta;
	theta = theta - hessian \ gradient;
end 