%locally_weighted_linear_regression to smooth the given signal

function[Y2] = locally_weighted_linear_regression(X, Y, tau)
n = length(X);
X = [ones(n,1), X];
Y2 = zeros(n,1);

for i = 1:n
	W = exp(-(X - X(i))) .^ 2 / (2 * tau^2);
	theta =  (X' * (X .* W))) \ (X' * (W .* Y));
	Y2(i) = X(i) * theta;
end 
