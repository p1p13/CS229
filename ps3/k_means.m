A = double(imread('mandrill-small.tiff'));
imshow(uint8(round(A)));
hold on;
figure;

k = 16;
X = zeros(16, 3);

for l = 1:k
	i = round(unifrnd(1, size(A,1)));
	j = round(unifrnd(1, size(A,2)));
	X(l, :) = (permute(A(i,j,:), [3 2 1]))';
end

U = X;
for iter = 1 : 200
	V = zeros(k, 3);
	total = zeros(k, 1);
	for i = 1 : size(A, 1)	
		for j = 1 : size(A, 2)
			dist = zeros(k, 1);
			for l = 1 : k
				d = U(l, :)' - permute(A(i, j, :), [3 2 1]);
				dist(l) = d' * d;
			end
			[value, cluster] = min(dist);
			total(cluster) += 1;
			V(cluster, :) += permute(A(i, j, :), [3 2 1])';
		end
	end
	for l = 1 : k
		if(total(l)> 0)
			V(l, :) /= total(l);
		end
	end

	U = V;
end	 

B = double(imread('mandrill-large.tiff'));
imshow(uint8(round(B)));
hold on;
figure;

B_COMP = B;
for i = 1 : size(B,1)
	for j = 1 : size(B,2)
		dist = zeros(k, 1);
		for l = 1 : k
			d = U(l, :)' - permute(B(i, j, :), [3 2 1]);
			dist(l) = d' * d;
		end
		[value, cluster] = min(dist);
		B_COMP(i, j, :) = permute(U(cluster, :), [1 3 2]);	
	end
end

imshow(uint8(round(B_COMP)));
