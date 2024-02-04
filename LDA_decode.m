function [outp, uncertainty, posteriors, bootgammas] = LDA_decode(t, train, test, p, lambda_var, lambda)






train_samples = squeeze(train.allsamples(t,:,:));
test_samples = squeeze(test.allsamples(t,:,:));
train_resp = train.resp;
Ntesttrials = size(test_samples,2);
D = size(train_samples,1);
N = size(train_resp, 1);


liks = zeros(Ntesttrials, p.nbinsstimval);
bootgammas = zeros(p.nboot,1);
for b=1:p.nboot
    %find W and Sigma
    set = randi(p.nsets);

    idx = randi(N,N,1);
    if p.shrinkmethod == "cv"
        while size(unique(train.cvid(idx)),1)<p.cvk
            idx = randi(N,N,1);
        end
    end

    boot_samples = train_samples(:,idx);
    boot_resp = train_resp(idx,:,set);
    %calculate sigma-tilde
    switch p.shrinkmethod

        case 'numeric'
            W = (boot_resp\boot_samples')';
            noise = boot_samples - (boot_resp*W')';
            [cov, gamma] = shrinkcov(noise);
            bootgammas(b) = gamma;
        case 'cv'
            cv_W{p.cvk} = [];
            cv_N{p.cvk} = [];
            cv_noise{p.cvk} = [];
            cv_pred{p.cvk} = [];
            boot_cvid = train.cvid(idx);
            boot_stimval = train.stimval(idx);
            for c = 1:p.cvk
                idxc = boot_cvid ~= c;
                cv_N{c} = size(boot_cvid(idxc), 1);
                cv_W{c} = (boot_resp(idxc,:)\boot_samples(:,idxc)')';
                cv_noise{c} = boot_samples(:,idxc) - (boot_resp(idxc,:)*cv_W{c}')';
                cv_pred{c} = p.basis_resp(:,:,set)*cv_W{c}';
            end
            errors = zeros(size(p.gammasearch, 2), p.cvk);
            for gam=1:size(p.gammasearch,2)
                gamma = p.gammasearch(gam);
                for c = 1:p.cvk
                    cov = shrinkcov(cv_noise{c},gamma);
                    try
                        prec_mat = invChol_mex(cov);
                    catch ME
                        if strcmp(ME.identifier, 'MATLAB:invChol_mex:dpotrf:notposdef')
                            fprintf('\nWARNING: Covariance estimate wasn''t positive definite. Trying again with another shrinkage parameter.\n');
                            errors(gam,c) = Inf;
                            break
                        else
                            rethrow(ME);
                        end
                    end

                    cvsamples = boot_samples(:,boot_cvid == c);
                    cvstimvals = boot_stimval(boot_cvid == c);
                    for cvj = 1:(N-cv_N{c})
                        samp = cvsamples(:,cvj);
                        res = bsxfun(@minus, samp', cv_pred{c});

                        ll = -0.5*diag(res*prec_mat*res');

                        probs = exp(ll-max(ll))';
                        probs = probs/sum(probs);
                        pop_vec = probs*exp(1i*p.binvals);
                        outp = angle(pop_vec); %Stimulus estimate (likelihood/posterior means)

                        errors(gam,c) = errors(gam,c) + abs(circ_dist(outp, cvstimvals(cvj)));
                    end
                end
            end
            overall_error = mean(errors, 2);
            [best_error, best_index] = min(overall_error);
            best_gamma = p.gammasearch(best_index);
            W = (boot_resp\boot_samples')';

            noise = boot_samples - (boot_resp*W')';
            cov = shrinkcov(noise, best_gamma);
            bootgammas(b) = best_gamma;

        case 'EM'
            W = (boot_resp\boot_samples')';
            noise = boot_samples - (boot_resp*W')';
            sampcov = noise*noise'/(N-1);
            cov = EM_cov(train.bestk(t), noise, sampcov);

        case 'TAFKAP'

            W = (boot_resp\boot_samples')';

            noise = boot_samples - (boot_resp*W')';
            cov = shrinkcov_TAFKAP(noise, lambda_var, lambda, W);

        case 'none'
            W = (boot_resp\boot_samples')';
            noise = boot_samples - (boot_resp*W')';
            cov = shrinkcov(noise, 0);
            bootgammas(b) = 0;
    end

    try
        prec_mat = invChol_mex(cov);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:invChol_mex:dpotrf:notposdef')
            fprintf('\nWARNING: Covariance estimate wasn''t positive definite. Trying again with another bootstrap sample.\n');
            continue
        else
            rethrow(ME);
        end
    end
    
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
pop_vec = posteriors*exp(1i*p.binvals); 
outp = angle(pop_vec); %Stimulus estimate (likelihood/posterior means)
uncertainty = sqrt(-2*log(abs(pop_vec)));
% outp2 = zeros(Ntesttrials, 1);
% uncertainty2 = zeros(Ntesttrials, 1);
% for i = 1:Ntesttrials
%     outp2(i) = mod(circ_mean(p.binvals, posteriors(i,:)',1), 2*pi);
%     uncertainty2(i) = sqrt(sum((circ_dist(p.binvals,outp(i))).^2 .* posteriors(i,:)'));
% end

    function [cov, gamma] = shrinkcov(noise,gamma)
        d = size(noise,1);
        n = size(noise,2);
        samplecov = noise*noise'/(n-1);
        nu = trace(samplecov)/d;
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
            gamma = n/((n-1)^2)*sum(varz, 'all')/(sum(s.^2, 'all') + sum((diag(samplecov)-nu).^2));
        end
        cov = (1-gamma)*samplecov + gamma*nu*eye(d);
    end


    function c = shrinkcov_TAFKAP(noise, lambda_var, lambda, W)
%         opt = optimoptions('lsqlin');
%         opt.Display = ('off');
        d = size(noise,1);
        n = size(noise,2);
        dia = tril(ones(d),-1)==1;

        samplecov = noise*noise'/(n-1);

        WWt = W*W';
        sigmasq = mean(noise'.^2);

        % vars = mean
        %OLS: how well can we predict covariance terms with some constant (coeff(2)) and
        %some coefficient associated with tuning-correlated noise
        %(coeff(1))?

            coeff = [WWt(dia), ones(sum(dia(:)),1)]\samplecov(dia);
        %     Constrain rho and sigma to be positive?
%         coeff = lsqlin([WWt(t), ones(sum(t(:)),1)], samplecov(t), [], [], [], [], [0 0], [], [], opt);

        targetcov = coeff(1)*WWt + coeff(2)*ones(d);

        targetdiag = lambda_var*median(sigmasq)+(1-lambda_var)*sigmasq;
        targetcov(eye(d)==1) = targetdiag;
        %
        %
        c = (1-lambda)*samplecov+lambda*targetcov;
        [~, pp] = chol(c);

        if pp>0
            [evec, eval] = eig(c);
            eval = diag(eval);
            min_eval = min(eval);
            eval = max(eval,1e-10);
            c = evec*diag(eval)/evec;
                    fprintf('\nWARNING: Non-positive definite covariance matrix detected. Lowest eigenvalue: %3.2g. Finding a nearby PD matrix by thresholding eigenvalues at 1e-10.\n', min_eval);
        end

    end

end

