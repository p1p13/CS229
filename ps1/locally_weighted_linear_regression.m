%locally_weighted_linear_regression to smooth the given signal

function[Y2] = locally_weighted_linear_regression(X, Y, tau)
n = length(X);
X2 = [ones(n,1), X];
Y2 = zeros(n,1);

for i = 1:n
	W = exp(-(X - X(i)) .^ 2 / (2 * tau^2));
	theta =  (X2' * (X2 .* W)) \ (X2' * (W .* Y));
	Y2(i) = X2(i, :) * theta;
	i;
end 
