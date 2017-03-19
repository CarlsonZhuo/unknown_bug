function [ g ] = logistic_grad( w, x_train, y_train, lambda, indices )
    e = exp(-1*y_train(indices).*(x_train(:,indices)'*w));
    s = e./(1+e);
    g = -(1/length(indices))*((s.*y_train(indices))'*x_train(:,indices)')';
    g = full(g) + lambda * w;
end