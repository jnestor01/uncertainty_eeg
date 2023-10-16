%ignoring random seeding for now

clear

%portion of data to hold out as testing data
p.test_part = 0.2;
%maximum number of bootstrap iterations
p.nboot = 1e3;
%number of basis functions to use for defining tuning curves
p.nchan = 8;
%number of sets of basis functions to choose from during bootstrapping
p.nsets = 4;
%exponent to which basis functions are raised
p.chan_exp = 5;
%number of bins to discretize possible stimulus values
p.nbinsstimval = 100;

%EEG specific (assuming 1000Hz data)
p.msperbin = 5;




%input: rows as time points columns as channels planes as trials
% samples = [];
% Ntrials = size(samples,3);
% Nelectrodes = size(samples,2);

%Bergen simulation function
%[samples, sp] = makeSNCData(struct('nvox', 50, 'ntrials', Ntrials, 'taumean', 0.7, 'ntrials_per_run', Ntesttrials, ...
%        'Wstd', 0.3, 'sigma', 0.3, 'randseed', p.randseed, 'shuffle_oris', 1, 'sim_stim_type', sim_stim_type, 'nclasses', nclasses));  

%Dummy data
samples =[];
Ntrials = 200;
Nelectrodes = 4;
stimval = rand(Ntrials,1) * 2*pi;
trialLength = 100;
ntsfactor = 1;
for l = 1:Nelectrodes
    pref = rand(1)*2*pi;
    for k = 1:trialLength
        samples(k,l,:) = abs(stimval-pref)/ntsfactor + randn(Ntrials,1);
    end
end

%cut off data at end if trial length is not divisible by binsize
NtimePoints = floor(size(samples,1)/p.msperbin);
samples = samples(1:NtimePoints*p.msperbin,:,:);
%time binning
samples = reshape(mean(reshape(samples,p.msperbin,[]),1),NtimePoints,Nelectrodes,Ntrials);

%If we treat each time point+electrode pair as a completely separate
%datapoint, with no assumptions about covariance, it's easiest to make the
%sample matrix 2d: rows are NtimePoints consecutive sets of Nelectrodes
samples = reshape(samples, (Nelectrodes*NtimePoints),Ntrials);

%Must index trials with vectors containing info about: true stimulus
%values, assignment to training or testing set, and run numbers (runs are
%used as the folds for cv later, other partitions can be used)

%true stimulus values are in [0,pi] for orientation but we have to scale to
%make it work right
% samples = ones(Ntrials,20);


Ntesttrials = floor(p.test_part*Ntrials);
Ntraintrials = Ntrials-Ntesttrials;
shuff = randperm(Ntrials);
test_trials = shuff(1:Ntesttrials)';
train_trials = shuff(Ntesttrials+1:end)';

train_samples = samples(:,train_trials);
test_samples = samples(:,test_trials);


binvals = linspace(0, 2*pi, p.nbinsstimval+1)';
binvals(end) = [];

basis_resp = zeros(p.nbinsstimval,p.nchan,p.nsets);

basis_prefs = (0:2*pi/(p.nchan):2*pi);
basis_prefs(end) = [];
basis_resp = nan(p.nbinsstimval, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        basis_resp(:,j,i) = max(0,cos(binvals - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^p.chan_exp);
    end
end

%plot(binvals, basis_resp(:,:,2))

train_resp = nan(Ntrials - Ntesttrials, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        train_resp(:,j,i) = max(0,cos(stimval(train_trials) - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^5);
    end
end
test_resp = nan(Ntesttrials, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        test_resp(:,j,i) = max(0,cos(stimval(test_trials) - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^5);
    end
end

%find_lambda placeholder
lambda = 0.5;
lambda_var = 0.5;

liks = zeros(Ntesttrials, p.nbinsstimval);

for b=1:p.nboot
%find W and Sigma
    set = randi(p.nsets);

    N = size(train_resp, 1);
    idx = randi(N,N,1);
    
    %calculate matrix and noise
    W = (train_resp(idx,:, set)\train_samples(:,idx)')';
    noise = train_samples(:,idx) - (train_resp(idx,:,set)*W')';

    %calculate sigma
    samplecov = noise*noise'/Ntraintrials;

    %come back to target covariance

    WWt = W*W';
    t = tril(ones(size(W,1)),-1)==1;

    sigmasq = mean(noise'.^2);

    % vars = mean
    %OLS: how well can we predict covariance terms with some constant (coeff(2)) and
    %some coefficient associated with tuning-correlated noise
    %(coeff(1))?
    coeff = [WWt(t), ones(sum(t(:)),1)]\samplecov(t);

    targetcov = coeff(1)*WWt + coeff(2)*ones(size(W,1));
    targetdiag = lambda_var*median(sigmasq)+(1-lambda_var)*sigmasq;
    targetcov(eye(size(W,1))==1) = targetdiag;
    % 
    % 
    % cov = (1-lambda)*samplecov+lambda*targetcov;
    cov = samplecov;

    % [~, flag] = chol(cov);
    % if flag>0
    %     [evec, eval] = eig(cov);
    %     eval = diag(eval);
    %     min_eval = min(eval);
    %     eval = max(eval,1e-10);
    %     C = evec*diag(eval)/evec;
    %     fprintf('\nWARNING: Non-positive definite covariance matrix detected. Lowest eigenvalue: %3.2g. Finding a nearby PD matrix by thresholding eigenvalues at 1e-10.\n', min_eval);
    % end

    prec_mat = invChol_mex(cov);
    %come back to ensuring covariance matrix is positive definite
    %cov matrix should always be positive definite given sufficient trial
    %number : data point ratio

    pred = basis_resp(:,:,set)*W';

    for j = 1:Ntesttrials
        samp = test_samples(:,j);
        res = bsxfun(@minus, samp', pred);

        %log-likelihood formula: ll(s) is -0.5 times a weighted sum of
        %squares of residuals for our predicted result at s, where weights are determined by inverting
        %variance (precision matrix) for each voxel
        %leave out precision matrix for now

        ll = -0.5*diag(res*prec_mat*res');

        probs = exp(ll-max(ll));
        probs = probs/sum(probs);

        liks(j,:) = liks(j,:) + probs';
    end


end

posteriors = bsxfun(@rdivide, liks, sum(liks,2));
uncertainty = var(posteriors,[],2);
[prob,idx] = max(posteriors,[],2);
outp = binvals(idx);
avgacc = mean(abs(outp-stimval(test_trials))/(2*pi));

