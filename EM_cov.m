%Derivation from Gregory Gundersen's "Factor Analysis in Detail"
%MLE_cov(k, noise, sampcov, lambda_0, smooth_alpha)
function [bestcov, lambda, psi, dev] = EM_cov(k, noise, sampcov, lambda_0, smooth_alpha)

if ~exist('smooth_alpha','var')
    smooth_alpha = 1;
end

if ~exist('lambda_0','var')
    princ = pca(sampcov);
    lambda_0 = princ(:,1:k);
end

p = size(noise, 1);
n = size(noise, 2);

lambda{1} = lambda_0;
psi{1} = eye(p).*sampcov - eye(p).*(lambda_0*lambda_0');
dev{1} = inf;
iter = 1;

while iter<100

    iter = iter+1;
    L = lambda{iter-1};
    P = psi{iter-1};

    beta = L' * inv(L*L' + P);
    Ezzt = n*(eye(k) - beta*L) + beta*(noise*noise')*beta';

    newL = (n*sampcov*beta')*inv(Ezzt);
    newP = eye(p).*((sampcov-newL*beta*sampcov));

    lambda{iter} = newL;
    psi{iter} = newP;

    dev{iter} = sum((lambda{iter}*lambda{iter}'+psi{iter}-sampcov).^2, 'all');

end

bestcov = lambda{iter}*lambda{iter}'+psi{iter};

%smoothing non-PD matrices
[evec, eval] = eig(bestcov);
eval = real(eval);
c = 10^(-smooth_alpha) * min(eval(eval>0));
newD = eye(p).*max(eval,c);
bestcov = real(evec)*newD*inv(real(evec));

end