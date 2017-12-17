% Program to smooth quassar data using locally weighted linear regression

load_quasar_data;
[m,n] = size(train_qso);

Y = train_qso(1, :)' ;
X = [ones(n,1), lambdas];

figure;
graph = plot(lambdas, train_qso(1, :), 'k+');
set(graph, 'linewidth', 1);
hold on;

taus = [1, 5, 10, 100, 1000];
symbols = { 'y+', 'm+', 'c+', 'r+', 'g+', 'b+' };

for i = 1:5
	tau = taus(i);
	Y2 = locally_weighted_linear_regression(lambdas, Y, tau);
	graph = plot(lambdas, Y2, symbols(i));
	set(graph, 'linewidth', 2);
end	

graph = legend('Raw data', 'tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000' )
set(graph, 'fontsize', 15)
