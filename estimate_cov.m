function c = estimate_cov(noise, lambda_var, lambda, W, samplecov)

    WWt = W*W';
    sigmasq = mean(noise'.^2);

    % vars = mean
    %OLS: how well can we predict covariance terms with some constant (coeff(2)) and
    %some coefficient associated with tuning-correlated noise
    %(coeff(1))?
    t = tril(ones(size(noise,1)),-1)==1;

    coeff = [WWt(t), ones(sum(t(:)),1)]\samplecov(t);

    targetcov = coeff(1)*WWt + coeff(2)*ones(size(W,1));
    targetdiag = lambda_var*median(sigmasq)+(1-lambda_var)*sigmasq;
    targetcov(eye(size(W,1))==1) = targetdiag;
    % 
    % 
    c = (1-lambda)*samplecov+lambda*targetcov;
    
end