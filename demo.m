clear;
% addpath('~/Google Drive/cookedData/');



%% Get the data X, y
%   1. X: [m*n], each column of X is one sample data;
%   2. y: [n*1], is the label of each sample data.A(i,:).
%   3. w: [m*1], is the number of features.



% load('../data/rcv1_train.binary.mat');
load('australian.mat');
% [y, X] = libsvmread('../data/ijcnn1.t');
X = X';
[d, n] = size(X);


% % Data preprocessing 
% X = [ones(size(X,1),1) X];
% [n, d] = size(X);
% X = X';
% sum1 = 1./sqrt(sum(X.^2, 1));
% if abs(sum1(1) - 1) > 10^(-10)
%     X = bsxfun(@rdivide,X,sqrt(sum(X.^2, 1)));
% end



%% Get the approximation of the best parameter
% lambda = 1/(n);
lambda = 1 / n;
Lmax   = (0.25*max(sum(X.^2,2)) + lambda);
number_of_data_passes = 50*2;
mb = 1;



% Declare functions

% Logistic
F_fgrad = @(w)logistic_grad(w,X,y,lambda,1:n);
F_pgrad = @(w,indices)logistic_grad(w,X,y,lambda,indices);
F_loss = @(w)sum(log(1+exp(-y.*(X'*w))))/n + 0.5*lambda*w'*w;

% Least Square
% Lmax   = (2*max(sum(X.^2,2)) + lambda);
% F_fgrad = @(w)X*(X'*w - y)/n + lambda*w;
% F_pgrad = @(w,Sample, Label)Sample*(Sample'*w - Label)/n + lambda*w * (mb/n);
% F_loss = @(w)0.5*(X'*w - y)'*(X'*w - y)/n + 0.5*lambda*w'*w;



% SVRG
w_SVRG = zeros(d, 1);
tic;
[histSVRG_l2, w_SVRG] = ...
        Alg_SVRG(X, y, ...
                F_loss, F_fgrad, F_pgrad, ...
                Lmax, number_of_data_passes*n, mb);

time_SVRG = toc;
fprintf('Time spent on SVRG: %f seconds \n', time_SVRG);
semilogy(histSVRG_l2 - min(histSVRG_l2))


