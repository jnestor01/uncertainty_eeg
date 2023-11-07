%ignoring random seeding for now

%portion of data to hold out as testing data
p.test_part = 0.2;
%maximum number of bootstrap iterations
p.nboot = 1e3;
%number of basis functions to use for defining tuning curves
%this parameter must be less than the number of unique stimulus values
%presented across trials
p.nchan = 6;
%number of sets of basis functions to choose from during bootstrapping
p.nsets = 4;
%exponent to which basis functions are raised
p.chan_exp = 3;
%number of bins to discretize possible stimulus values
p.nbinsstimval = 100;
%size of time bins
p.msperbin = 1;

% clear
runs = [1:8];
% meg = load('R1507_CupcakeAperture_4.25.19_ebci_condData.mat');
timewindow = [1101:1501];
targetchannels = [1:157];
trialsPerRun = size(meg.D.condData,3);

lambda_range = linspace(0,1,50);
lambda_var_range = linspace(0,1,4);
%stim onset, not used in decoding just for making strings
t0 = 0;

%% MEG preprocess%%
allsamples = zeros(numel(timewindow), numel(targetchannels), trialsPerRun*numel(runs));
cvid = zeros(trialsPerRun*numel(runs),1);
stimval = cvid;
Ntrials = 0;
for r=1:numel(runs)
    ss = meg.D.condData(timewindow, targetchannels, :, r);
    allsamples(:,:,(Ntrials+1:Ntrials+trialsPerRun)) = ss;
    cvid(Ntrials+1:Ntrials+trialsPerRun) = r;
    trials = load(sprintf('R1507_20190425_run0%d_CupcakeAperture_20190425.mat', r));
    for i=1:trialsPerRun
        stimval(Ntrials+i) = trials.expt.trialsPresented(i).orientation;
    end

    [x,y,z] = ind2sub(size(ss), find(isnan(ss)));
    removeTrials = unique(z);
    allsamples(:,:,removeTrials+Ntrials) = [];
    cvid(removeTrials+Ntrials) = [];
    stimval(removeTrials+Ntrials) = [];

    Ntrials = Ntrials + trialsPerRun-numel(removeTrials);
end
%%

% Normalization options
% samples = samples/max(abs(max(samples, [], 'all')), abs(min(samples, [], 'all')));
% samples = samples/mean(abs(samples),'all');

Nelectrodes = size(allsamples,2);

stimval = stimval/180 * 2 * pi;


%input: rows as time points columns as channels planes as trials
% samples = [];
% Ntrials = size(samples,3);
% Nelectrodes = size(samples,2);


%% Simplest dummy data%%
% samples =[];
% Ntrials = 600;
% Nelectrodes = 4;
% stimval = rand(Ntrials,1) * 2*pi;
% trialLength = 100;
% signalstrength = 10;
% noisestrength = 0.1;
% for l = 1:Nelectrodes
%     pref = rand(1)*2*pi;
%     for k = 1:trialLength
%         samples(k,l,:) = abs(stimval-pref)*signalstrength + randn(Ntrials,1)*noisestrength;
%     end
% end
% trialsPerCV = 100;
% cvid = repmat([1:Ntrials/trialsPerCV]',trialsPerCV,1);
%%


%% TAFKAP dummy data
% [samples, sp] = makeSNCData(struct('nvox', 50, 'ntrials', Ntrials, 'taumean', 0.7, 'ntrials_per_run', Ntesttrials, ...
%     'Wstd', 0.3, 'sigma', 0.3, 'randseed', 1234, 'shuffle_oris', 1, 'sim_stim_type', 'circular', 'nclasses', nclasses));
% cvid=sp.run_idx;
% stimval=sp.stimval;
% samples = samples';
% Nelectrodes = size(samples,2);
%%

% %% TAFKAP ideal time series data
% Ntrials = 600; 
% trialsPerCV = 100;
% Nelectrodes = 15;
% timewindow = 1:101;
% allsamples = zeros(size(timewindow,2)+1,Nelectrodes,Ntrials);
% t0 = 20;
% 
% tuning_corr_noise = 15;
% ind_noise = 4;
% global_noise = 10;
% decay = 0.75;
% 
% signalstrength = 5;
% 
% stimval = rand(Ntrials,1) * 2*pi;
% basis_prefs = (0:2*pi/(p.nchan):2*pi);
% basis_prefs(end) = [];
% pref = rand(1,Nelectrodes) * 2 *pi;
% 
% pref_similarity = zeros(Nelectrodes, p.nchan);
% for n = 1:Nelectrodes
%     pref_similarity(n,:) = pi - abs(circ_dist(pref(n), basis_prefs));
% end
% 
% for n = 1:Ntrials
%     for t = 2:t0
%         tunenoise = randn(p.nchan,1)*tuning_corr_noise;
%         for e = 1:Nelectrodes
%             allsamples(t,e,n) = dot(pref_similarity(e,:),tunenoise) + randn(1)*ind_noise;
%         end
%         allsamples(t,:,n) = allsamples(t,:,n) + global_noise*randn(1,Nelectrodes) + decay*allsamples(t-1,:,n);
%     end
%     for t = t0+1:size(timewindow,2)+1
%         tunenoise = randn(p.nchan,1)*tuning_corr_noise;
%         for e = 1:Nelectrodes
%             allsamples(t,e,n) = signalstrength*(pi - abs(circ_dist(pref(e),stimval(n)))^2) + dot(pref_similarity(e,:),tunenoise) + randn(1)*ind_noise;
%         end
%         allsamples(t,:,n) = allsamples(t,:,n) + global_noise*randn(1,Nelectrodes) + decay*allsamples(t-1,:,n);
%     end
% end
% allsamples(1,:,:) = [];
% cvid = repmat([1:Ntrials/trialsPerCV]',trialsPerCV,1);
% %%


% cut off data at end if trial length is not divisible by binsize
NtimePoints = floor(size(allsamples,1)/p.msperbin);
allsamples = allsamples(1:NtimePoints*p.msperbin,:,:);
%time binning
allsamples = reshape(mean(reshape(allsamples,p.msperbin,[]),1),NtimePoints,Nelectrodes,Ntrials);

%If we treat each time point+electrode pair as a completely separate
%datapoint, with no assumptions about covariance, it's easiest to make the
%sample matrix 2d: rows are NtimePoints consecutive sets of Nelectrodes
% samples = reshape(samples, (Nelectrodes*NtimePoints),Ntrials);

%Must index trials with vectors containing info about: true stimulus
%values, assignment to training or testing set, and run numbers (runs are
%used as the folds for cv later, other partitions can be used)

%true stimulus values are in [0,pi] for orientation but we have to scale to
%make it work right
% samples = ones(Ntrials,20);

%it's a surprise tool that will help us later
% t = tril(ones(size(samples,1)),-1)==1;
% opt = optimoptions('lsqlin');
% opt.Display = ('off');

binvals = linspace(0, 2*pi, p.nbinsstimval+1)';
binvals(end) = [];

basis_prefs = (0:2*pi/(p.nchan):2*pi);
basis_prefs(end) = [];
basis_resp = nan(p.nbinsstimval, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        basis_resp(:,j,i) = max(0,cos(binvals - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^p.chan_exp);
    end
end






Ntesttrials = floor(p.test_part*Ntrials);
Ntraintrials = Ntrials-Ntesttrials;
shuff = randperm(Ntrials);
test_trials = shuff(1:Ntesttrials)';
train_trials = shuff(Ntesttrials+1:end)';

resp = nan(Ntrials, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        resp(:,j,i) = max(0,cos(stimval - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^5);
    end
end

train_resp = resp(train_trials,:,:);
test_resp = resp(test_trials,:,:);

cvid(test_trials) = 0;

cv_folds = unique(cvid(cvid ~= 0));
k = length(cv_folds);


timestr = strings(NtimePoints,1);
if p.msperbin==1
    for m = 1:NtimePoints
        timestr(m) = sprintf('t%d', timewindow(m)-t0);
    end
else
    tmat = (timewindow(1):p.msperbin:timewindow(end)) - t0;
    for m = 1:NtimePoints
        timestr(m) = sprintf('t%d - t%d', tmat(m), tmat(m+1));
    end
end


postmat = zeros(Ntesttrials, p.nbinsstimval, NtimePoints);
uncmat = zeros(NtimePoints,1);
errmat = zeros(NtimePoints,1);

%START TIME POINT LOOP
for m = 1:NtimePoints

    samples = allsamples(m,:,:);
    samples = reshape(samples, Nelectrodes, Ntrials);


train_samples = samples(:,train_trials);
test_samples = samples(:,test_trials);




% plot(binvals, basis_resp(:,:,2))

%for each trial how do we expect each basis population to respond


%cross-validation over cvid trials to optimize lambda and lambda_var


cv_W{k} = [];
c_noise{k} = [];
t_noise{k} = [];
valsamplecov{k} = [];
c_samplecov{k} = [];

for j = 1:k

    set = 1;
    idj = cvid==cv_folds(j);
    idjc = ~idj;
    idjc(test_trials) = 0;
    c_resp = resp(idjc, :, set);
    c_samples = samples(:,idjc);    
    t_resp = resp(idj, :, set);
    t_samples = samples(:,idj);
    cv_W{j} = (c_resp\c_samples')';
    c_noise{j} = c_samples - (c_resp*cv_W{j}')';
    t_noise{j} = t_samples - (t_resp*cv_W{j}')';
    valsamplecov{j} = (t_noise{j}*t_noise{j}')/size(t_noise{j},2);
    c_samplecov{j} = (c_noise{j}*c_noise{j}')/size(c_noise{j},2);

end

ranges{1} = lambda_var_range;
ranges{2} = lambda_range;

s = cellfun(@length, ranges);
Ngrid = min(max(2, ceil(sqrt(s))), s); %Number of values to visit in each dimension (has to be at least 2, except if there is only 1 value for that dimension)

grid_vec = cellfun(@(x,y) linspace(1, y, x), num2cell(Ngrid), num2cell(s), 'UniformOutput', 0);
[grid_x, grid_y] = meshgrid(grid_vec{1}, grid_vec{2});
[grid_l1, grid_l2] = meshgrid(ranges{1}, ranges{2});
sz = fliplr(cellfun(@numel, ranges));

losses = nan(numel(grid_x),1);
    x={};

for grid_iter=1:numel(grid_x)
    this_lambda = [ranges{1}(grid_x(grid_iter)) ranges{2}(grid_y(grid_iter))];
    loss = 0;
    infs = 0;
    for cv_iter2=1:k
        valC = valsamplecov{cv_iter2}; %sample covariance of validation data
        estC = estimate_cov_blankertz(c_noise{cv_iter2}, this_lambda(2), c_samplecov{cv_iter2});
        lossk = cov_loss(estC, valC);
        if grid_iter==10
        x{cv_iter2,1}=valC;
        x{cv_iter2,2}=estC;
        end
        if isinf(lossk)
            infs = infs+1;
        else
            loss = loss + lossk;
        end
    end
    if (k-infs)<=(k/2)
        loss = inf;
    else
        loss = loss/(k-infs);
    end
    losses(grid_iter) = loss;
%     fprintf('\n %02d/%02d -- lambda_var: %3.2f, lambda: %3.2f, loss: %5.4g', [grid_iter, numel(grid_x), this_lambda, losses(grid_iter)]);
end

[best_loss, best_idx] = min(losses);
visited = sub2ind(sz,grid_y,grid_x); visited = visited(:);
best_idx = visited(best_idx);

fprintf('\n--PATTERN SEARCH--');        
step_size = 2^floor(log2(diff(grid_y(1:2)/2))); 

while 1
    [best_y,best_x] = ind2sub(sz, best_idx);
    new_x = best_x + [-1 1 -1 1]'*step_size;
    new_y = best_y + [-1 -1 1 1]'*step_size;
    del_idx = new_x<=0 | new_x> numel(ranges{1}) | new_y<=0 | new_y > numel(ranges{2});
    new_x(del_idx) = []; new_y(del_idx) = [];
    new_idx = sub2ind(sz, new_y, new_x);
    new_idx = new_idx(~ismember(new_idx, visited));

    if ~isempty(new_idx)
        this_losses = nan(size(new_idx));
        for ii = 1:length(new_idx)
            this_lambda = [grid_l1(new_idx(ii)), grid_l2(new_idx(ii))];
            loss = 0;
            infs = 0;
            for cv_iter2=1:k
                valC = valsamplecov{cv_iter2}; %sample covariance of validation data
                estC = estimate_cov_blankertz(c_noise{cv_iter2}, this_lambda(2), c_samplecov{cv_iter2});
                lossk = cov_loss(estC, valC);
                if isinf(lossk)
                    infs = infs+1;
                else
                    loss = loss + lossk;
                end
            end
            if (k-infs)<=(k/2)
                loss = inf;
            else
                loss = loss/(k-infs);
            end
            this_losses(ii) = loss;
            fprintf('\nStep size: %d, lambda_var: %3.2f, lambda: %3.2f, loss: %5.4g', [step_size, this_lambda, this_losses(ii)]);
        end
        visited = vertcat(visited, new_idx);
        losses = vertcat(losses, this_losses);
    end
    if any(this_losses<best_loss)
        [best_loss, best_idx] = min(losses);
        best_idx = visited(best_idx);
    elseif step_size>1
        step_size = step_size/2;
    else
        break
    end
end
lambda_var = grid_l1(best_idx); lambda = grid_l2(best_idx);

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

    cov = estimate_cov_blankertz(noise, lambda, samplecov);
%     cov = samplecov;

    % [~, flag] = chol(cov);
    % if flag>0
    %     [evec, eval] = eig(cov);
    %     eval = diag(eval);
    %     min_eval = min(eval);
    %     eval = max(eval,1e-10);
    %     C = evec*diag(eval)/evec;
    %     fprintf('\nWARNING: Non-positive definite covariance matrix detected. Lowest eigenvalue: %3.2g. Finding a nearby PD matrix by thresholding eigenvalues at 1e-10.\n', min_eval);
    % end

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
% outp = circ_mean(posteriors,[],2);
outp = zeros(Ntesttrials, 1);
uncertainty = zeros(Ntesttrials, 1);
for i = 1:Ntesttrials
    outp(i) = mod(circ_mean(binvals, posteriors(i,:)',1), 2*pi);
    uncertainty(i) = sqrt(sum((circ_dist(binvals,outp(i))).^2 .* posteriors(i,:)'));
end
errors = abs(circ_dist(outp, stimval(test_trials)));
avgerrort = mean(errors)/(2*pi);
avgunct = mean(uncertainty);
errmat(m) = avgerrort;
uncmat(m) = avgunct;
fprintf('\n t=%s, error=%3.2f, uncertainty=%3.2f', timestr(m), avgerrort, avgunct);
postmat(:,:,m) = posteriors; 

end

avgerror = mean(errmat);
avgunc = mean(uncmat);

pooledpost = mean(postmat, 3);
pooledoutp = zeros(Ntesttrials, 1);
pooleduncertainty = zeros(Ntesttrials, 1);
for i = 1:Ntesttrials
    pooledoutp(i) = mod(circ_mean(binvals, pooledpost(i,:)',1), 2*pi);
    pooleduncertainty(i) = sqrt(sum((circ_dist(binvals,pooledoutp(i))).^2 .* pooledpost(i,:)'));
end
poolederrors = abs(circ_dist(pooledoutp, stimval(test_trials)));
avgpoolederror = mean(poolederrors)/(2*pi);
avgpooledunc = mean(pooleduncertainty);


plotoptions = gca;

plottrial = 8;
plot(binvals/2,pooledpost(plottrial,:));
title('Pooled posterior for one test trial');
xlabel('orientation');
ylabel('probability');
xline(stimval(test_trials(plottrial))/2,'--','stimulus orientation');

col = ["red", "blue"];
plottrial = 5;
clear plo
for m=1:t0
pl = plot(binvals/2, postmat(plottrial,:,m), 'color', col(1));
plo(m) = pl(1); hold on
end
for m=t0+1:timewindow(end)
pl = plot(binvals/2, postmat(plottrial,:,m), 'color', col(2)); 
plo(m) = pl(1); hold on
end
title('Posteriors at all time points for one test trial');
xlabel('orientation');
ylabel('probability');
xline(stimval(test_trials(plottrial))/2,'--','stimulus orientation');
legend([plo(1), plo(t0+1)],'before stimulus onset', 'after stimulus onset');

plot(uncmat);
xticklabels(timestr(1:2:end));
xticks(1:2:timewindow(end)/p.msperbin);
xlabel('time window');
ylabel('decoded uncertainty');
xline(t0/p.msperbin, '--', 'stimulus onset');


plot(errmat);
xticklabels(timestr(1:2:end));
xticks(1:2:timewindow(end)/p.msperbin);
xlabel('time window');
ylabel('percent decoding error');
yline(0.25, '--', 'chance');
xline(t0/p.msperbin, '--', 'stimulus onset');

col = ['red', 'blue'];
pre = scatter(errmat(1:t0),uncmat(1:t0), 'Color', col(1)); hold on
post = scatter(errmat(t0+1:end), uncmat(t0+1:end), 'Color', col(2));
ylabel('decoded uncertainty');
xlabel('percent decoding error');
legend([pre(1), post(1)], 'before stimulus onset', 'after stimulus onset');

function c = estimate_cov(noise, lambda_var, lambda, W, samplecov)
opt = optimoptions('lsqlin');
opt.Display = ('off');
t = tril(ones(size(noise,1)),-1)==1;



    WWt = W*W';
    sigmasq = mean(noise'.^2);

    % vars = mean
    %OLS: how well can we predict covariance terms with some constant (coeff(2)) and
    %some coefficient associated with tuning-correlated noise
    %(coeff(1))?

%     coeff = [WWt(t), ones(sum(t(:)),1)]\samplecov(t);
%     Constrain rho and sigma to be positive?
    coeff = lsqlin([WWt(t), ones(sum(t(:)),1)], samplecov(t), [], [], [], [], [0 0], [], [], opt);

    targetcov = coeff(1)*WWt + coeff(2)*ones(size(W,1));

    targetdiag = lambda_var*median(sigmasq)+(1-lambda_var)*sigmasq;
    targetcov(eye(size(W,1))==1) = targetdiag;
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
%         fprintf('\nWARNING: Non-positive definite covariance matrix detected. Lowest eigenvalue: %3.2g. Finding a nearby PD matrix by thresholding eigenvalues at 1e-10.\n', min_eval);
    end
    
end

function loss = cov_loss(est_cov, samp_cov)
        try
            loss = (logdet(est_cov, 'chol') + sum(sum(invChol_mex(est_cov).*samp_cov)))/size(samp_cov,2);
        catch ME
            if any(strcmpi(ME.identifier, {'MATLAB:posdef', 'MATLAB:invChol_mex:dpotrf:notposdef'}))
                loss = (logdet(est_cov) + trace(est_cov\samp_cov))/size(samp_cov,2);
            else
                rethrow(ME);
            end
        end
        if imag(loss)~=0, loss = inf; end
end

function c = estimate_cov_blankertz(noise, lambda, samplecov)
    
    n = size(samplecov,1);
    upsilon = trace(samplecov)/n;
    c = (1-lambda)*samplecov + lambda*upsilon*eye(n);
    
end
