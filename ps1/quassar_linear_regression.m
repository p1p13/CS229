%Linear regression of quassar training data

load_quasar_data;
[m,n] = size(train_qso);

Y = train_qso(1, :)' ;
X = [ones[=(n,1), lambdas];
theta = (X' * X) \ (X' * Y);

figure;
graph = plot(lambdas, train_qso(1, :), 'k+');
set(graph, 'linewidth', 1);
hold on;
graph = plot(lambdas, X * theta, 'r-');
set(graph, 'linewidth', 2);

leg = legend('raw data', 'regression line');
set(leg, 'fontsize', 20);