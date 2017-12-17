% Functional Regression

% Smooth all quasar training sets

load_quasar_data;
[m, n] = size(train_qso);
[m2, n2] = size(test_qso);
tau = 5;

for j = 1:m
	Y = train_qso(j,:)';
	train_qso(j,:) = locally_weighted_linear_regression(lambdas, Y, tau)';
end

for j = 1:m2
	Y = test_qso(j,:)';
	test_qso(j,:) = locally_weighted_linear_regression(lambdas, Y, tau)';
end

train_right = train_qso(:, 151:end);
train_left = train_qso(:, 1:50);
test_right = test_qso(:, 151:end);
test_left = test_qso(:, 1:50);

distance = zeros(m,m);
for i = 1:m
	for j = 1 : m
		distance(i, j) = norm(train_right(i, :) - train_right(j, :)) ^ 2;
	end
end
distance = distance / max(distance(:));

f_left = zeros(m, 50);
k = 3;

for i = 1:m
	[sorted_distance, indices] = sort(distance(:,i), 1, 'ascend');
	k_neighbours = ones(m,1);
	k_neighbours(indices(k+1:end)) = 0;
	kernel = max(1 - distance(:, i), 0);
	kernel = kernel .* k_neighbours;
	f_left(i, :) = train_left' * kernel / sum(kernel);
end

% training error
training_error = sum((train_left(:) - f_left(:)) .^ 2);
training_error /= m;
fprintf(1, 'Average trainign error: %1.4f\n', training_error);