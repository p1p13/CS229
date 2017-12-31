function [ind, thresh] = find_best_threshold(X, y, p_dist)
% FIND_BEST_THRESHOLD Finds the best threshold for the given data
%
% [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
%   thresh and index ind that gives the best thresholded classifier for the
%   weights p_dist on the training data. That is, the returned index ind
%   and threshold thresh minimize
%
%    sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
%
%   OR
%
%    sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
%
%   We must check both signed directions, as it is possible that the best
%   decision stump (coordinate threshold classifier) is of the form
%   sign(threshold - x_j) rather than sign(x_j - threshold).
%
%   The data matrix X is of size m-by-n, where m is the training set size
%   and n is the dimension.
%
%   The solution version uses efficient sorting and data structures to perform
%   this calculation in time O(n m log(m)), where the size of the data matrix
%   X is m-by-n.

[mm, nn] = size(X);
ind = 1;
thresh = 0;
% ------- Your code here -------- %
%
% A few hints: you should loop over each of the nn features in the X
% matrix. It may be useful (for efficiency reasons, though this is not
% necessary) to sort each coordinate of X as you iterate through the
% features.
err = inf;

for j = 1:nn
	[x_sort, ids] = sort(X(:,j), 1, 'descend');
	p_sort = p_dist(ids);
	y_sort = y(ids);
	thresh_list = (x_sort + circshift(x_sort,1)) / 2;
	thresh_list(1) = x_sort(1) + 1;
	inc_list = circshift(p_sort .* y_sort, 1);
	inc_list(1) = 0;
	error = ones(mm, 1) * (p_sort' * (y_sort == 1));
	error = error + cumsum(inc_list);
	[min_error, id] = min(error);	
	[max_error, max_id] = max(error);
	max_error = 1 - max_error;
	curr_error =  min(min_error, max_error);
	if(min_error > max_error)
		id = max_id;
	end
	if(curr_error < err)
		ind = j;
		thresh = thresh_list(id);
		err = curr_error;
	end
end