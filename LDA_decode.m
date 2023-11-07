function [outp, uncertainty, posteriors] = LDA_decode(t, train, test, p)


train_samples = squeeze(train.allsamples(t,:,:));
test_samples = squeeze(test.allsamples(t,:,:));
train_resp = train.resp;
Ntesttrials = size(test_samples,2);

liks = zeros(Ntesttrials, p.nbinsstimval);

for b=1:p.nboot
    %find W and Sigma
    set = randi(p.nsets);

    N = size(train_resp, 1);
    idx = randi(N,N,1);

   %calculate weight vectors independently for each channel?

    
    W = (train_resp(idx,:, set)\train_samples(:,idx)')';
    noise = train_samples(:,idx) - (train_resp(idx,:,set)*W')';

    %calculate sigma-tilde
    cov = shrinkcov(noise);

    try
        prec_mat = invChol_mex(cov);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:invChol_mex:dpotrf:notposdef')
            fprintf('\nWARNING: Covariance estimate wasn''t positive definite. Trying again with another bootstrap sample.\n');
            continue
        else
            rethrow(ME);
        end
    end    %come back to ensuring covariance matrix is positive definite
    %sample cov matrix should always be positive definite given sufficient trial
    %number : data point ratio
    
    %based on our estimate of W, how would we expect our data to look given
    %each possible stimulus value?
    pred = p.basis_resp(:,:,set)*W';


    for j = 1:Ntesttrials
        samp = test_samples(:,j);
        res = bsxfun(@minus, samp', pred);

        %log-likelihood formula: ll(s) is -0.5 times a weighted sum of
        %squares of residuals for our predicted result at s, where weights are determined by inverting
        %variance (precision matrix) for each voxel

        ll = -0.5*diag(res*prec_mat*res');

        probs = exp(ll-max(ll));
        probs = probs/sum(probs);

        liks(j,:) = liks(j,:) + probs';
    end


end

posteriors = bsxfun(@rdivide, liks, sum(liks,2));
% outp = circ_mean(posteriors,[],2);
outp = zeros(Ntesttrials, 1);
uncertainty = zeros(Ntesttrials, 1);
for i = 1:Ntesttrials
    outp(i) = mod(circ_mean(p.binvals, posteriors(i,:)',1), 2*pi);
    uncertainty(i) = sqrt(sum((circ_dist(p.binvals,outp(i))).^2 .* posteriors(i,:)'));
end

    function cov = shrinkcov(noise,gamma)
        d = size(noise,1);
        n = size(noise,2);
        samplecov = noise*noise'/(n-1);
        upsilon = trace(samplecov)/d;
        if nargin<2

            mu = mean(noise,2);
            x = bsxfun(@minus, noise, mu);
            z = zeros(d,d,n);

            for w = 1:n
                z(:,:,w) = x(:,w)*x(:,w)';
            end
            varz = var(z,0,[1 2]);
            s = samplecov;
            s(eye(d)==1) = 0;
            gamma = n/((n-1)^2)*sum(varz, 'all')/(sum(s.^2, 'all') + sum((diag(samplecov)-upsilon).^2));
        end
        cov = (1-gamma)*samplecov + gamma*upsilon*eye(d);
    end

end

