function [hist, x] = Alg_SVRG(samples, labels, F_loss, F_fgrad, F_pgrad, Lmax, max_it, mb)

if nargin < 6
    mb = 10;  
end 

% initialization
[d, N] = size(samples);
x_tilde= zeros(d,1);
x      = zeros(d,1);
m      = fix(2*N/mb);
mu     = zeros(d,1); 
max1   = fix(max_it/(2*N))+1;
step_s  = 1 / (10*Lmax);
hist   = zeros(max1+1,1);

for k = 1:max1
    fprintf('current iteration:%d, current loss: %f\n', k , F_loss(x_tilde));
    hist(k) = F_loss(x_tilde);

    rnd_pm = [randperm(N)];
    mu = F_fgrad(x_tilde);

    for j = 1:m 
        %%% randomly choose minibatch   
        idx = j;
        if idx <= N-mb+1
            ix = idx:idx+mb-1;
        else
            ix = [1:(idx+mb-N-1), idx:N];
        end
        I = sort(rnd_pm(ix));    

        % Gradient   
        gg = F_pgrad(x, I) - F_pgrad(x_tilde, I);
        gg = gg/mb + mu;

        x = x - step_s * gg;
    end 
    x_tilde = x;
end
hist(max1+1) = F_loss(x_tilde);

end