function [bestcov, likelihood, gammas] = MLE_cov(noise, kmin, kmax, shrink, l)
% kmin = 1;
% kmax = 30;
% l-fold cross-validation

likelihood = zeros(1,kmax-kmin+1);
gammas = zeros(1,kmax-kmin+1);

p = size(noise,1);
n = size(noise,2);

idx = randperm(n);
trialsPerFold = floor(n/l);

%find best k
for fold = 1:l
    leaveout = idx(trialsPerFold*(fold-1)+1:trialsPerFold*fold);
    noise_fold{fold} = noise;
    noise_fold{fold}(:,leaveout) = [];
    sampcov_fold{fold} = noise_fold{fold}*noise_fold{fold}'/(n-trialsPerFold-1);
    test_cov{fold} = noise(:,leaveout)*noise(:,leaveout)'/(trialsPerFold-1);
for k = kmin:kmax
    if k==p
       [bestcov{k,fold}, b, c, d] = EM_cov(k,noise_fold{fold},sampcov_fold{fold},sampcov_fold{fold});
    else
    [bestcov{k,fold}, b, c, d] = EM_cov(k,noise_fold{fold},sampcov_fold{fold});
    end


    
    %shrinkage of the diagonal
    nu = trace(bestcov{k, fold})/p;
    mu = mean(noise_fold{fold},2);
    x = bsxfun(@minus, noise_fold{fold}, mu);
    z = zeros(p,p,n-trialsPerFold);
    for w = 1:n-trialsPerFold
        z(:,:,w) = x(:,w)*x(:,w)';
    end
    varz = var(z,0,[1 2]);
    s = bestcov{k, fold};
    s(eye(p)==1) = 0;

    if shrink == 1
    gamma = n/((n-1)^2)*sum(varz, 'all')/(sum(s.^2, 'all') + sum((diag(bestcov{k, fold})-nu).^2));
    bestcov{k, fold} = (1-gamma)*bestcov{k, fold} + gamma*nu*eye(p);
    gammas(k) = gammas(k)+gamma;    
    end
    likelihood(k) = likelihood(k)-trace(inv(bestcov{k, fold})*test_cov{fold}) - log(det(bestcov{k, fold}));
    if ~isreal(likelihood(k))||isinf(likelihood(k))
        likelihood(k) = -inf;
    end
end
end
likelihood = likelihood/l;    
gammas = gammas/l;
end