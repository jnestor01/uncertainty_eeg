%Kok et al. 2017 LDA decoding method with the differences being the use of bootstrapping and a distribution of
%likelihoods is computed instead of a point estimate

%compared to TAFKAP, this method does not give particular attention to
%tuning-correlated noise
% warning('off', 'MATLAB:rankDeficientMatrix');

%portion of data to hold out as testing data
p.test_part = 0.1;
%maximum number of bootstrap iterations
p.nboot = 10^3;
%number of basis functions to use for defining tuning curves
%this parameter must be less than the number of unique stimulus values
%presented across trials
%unless you compute weights separately
p.nchan = 8;
%number of sets of basis functions to choose from during bootstrapping
p.nsets = 4;
%exponent to which basis functions are raised
p.chan_exp = 5;
%number of bins to discretize possible stimulus values
p.nbinsstimval = 100;
%size of time bins
p.msperbin = 10;

%shrinkage of the diagonal toward its mean by a constant gamma
%estimate optimal shrinkage parameter numerically or by cross-validation
p.shrinkmethod = 'numeric';
%p.shrinkmethod = 'cv';

%factor analysis method: estimate covariance by a low-rank matrix L*L' plus
%a diagonal matrix D.
%find L of rank k by expectation-maximization where best k is estimated
%for each time point
% p.shrinkmethod = 'EM';

%shrinkage toward W*W'+D where W is the weight matrix of basis functions on
%channels. theoretically accounts for tuning-correlated noise
%p.shrinkmethod = 'TAFKAP';

% %p.shrinkmethod = 'none';
% p.kmin = 1;
% p.kmax = 30;

p.cvk = 6;
%only for cv
p.gammasearch = linspace(0,1,50);

%regularization options in order that they are done
p.doprestimulussubtraction = 0;
%number of ms before stimulus onset to average then subtract (each sensor
%each trial)
p.subtractiontimewindow = 200;
p.dosensordemean = 0;
p.dosensorzscore = 0;
p.dopatternzscore = 1;


oris = [1:180];
% eeg = load('cupcake0001_most_basic_preprocess.mat');
trials = load('cupcake0001.mat');
timewindow = [800:1400];
targetchannels = [1:63];
Nchan = size(targetchannels,2);

%stim onset, not used in decoding just for making strings
t0 = 1000;

%% EEG preprocess%%
%for cv estimation of optimal shrinkage parameter
% cvid = Shuffle(repmat([1:p.cvk]', ceil(fullN/p.cvk), 1));
% want approximately equal trials of each orientation in each cv fold
% cvid = repmat([1:p.cvk]', ceil(fullN/p.cvk), 1);
%chan x time x trial
%want time x chan x trial

Ntrials = size(eeg.outp.data, 3);
allsamples = eeg.outp.data(:,timewindow,:);
allsamples(64,:,:) = [];
allsamples = permute(allsamples, [2,1,3]);
stimval = trials.expt.trials(:,1);
contrast = trials.expt.trials(:,3);
fixes = trials.expt.trials(:,7);
removeTrials = fixes<1;
allsamples(:,:,removeTrials) = [];
stimval(removeTrials) = [];
contrast(removeTrials) = [];
Ntrials = Ntrials - sum(removeTrials);


%time binning
% cut off data at end if trial length is not divisible by binsize
NtimePoints = floor(size(allsamples,1)/p.msperbin);
allsamples = allsamples(1:NtimePoints*p.msperbin,:,:);
allsamples = reshape(mean(reshape(allsamples,p.msperbin,[]),1),NtimePoints,Nchan,Ntrials);

if p.dopatternzscore
    means = mean(allsamples,2);
    sds = std(allsamples, [], 2);
    allsamples = bsxfun(@minus, allsamples, means);
    allsamples = bsxfun(@rdivide, allsamples, sds);
end

stimval = stimval/180 * 2 * pi;

%% Channel response precompute %%
p.binvals = linspace(0, 2*pi, p.nbinsstimval+1)';
p.binvals(end) = [];

basis_prefs = (0:2*pi/(p.nchan):2*pi);
basis_prefs(end) = [];
p.basis_resp = nan(p.nbinsstimval, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        p.basis_resp(:,j,i) = max(0,cos(p.binvals - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^p.chan_exp);
        p.basis_resp(:,j,i) = p.basis_resp(:,j,i) - mean(p.basis_resp(:,j,i));

    end
end

%The basis responses are demeaned s.t. their average over
%trials equals zero for each basis channel
resp = nan(Ntrials, p.nchan,p.nsets);
for i = 1:p.nsets
    for j = 1:p.nchan
        resp(:,j,i) = max(0,cos(stimval - (basis_prefs(j)+((i-1)*(2*pi/p.nchan/p.nsets)))).^5);
        resp(:,j,i) = resp(:,j,i) - mean(resp(:,j,i));
    end
end

Ntesttrials = floor(p.test_part*Ntrials/2)*2;
Ntraintrials = Ntrials-Ntesttrials;
shuff = randperm(Ntrials);
test.trials = shuff(1:Ntesttrials)';
train.trials = shuff(Ntesttrials+1:end)';
%Test split to get an equal number of high and low contrast trials in each
%set
shuff = 1:Ntrials;
shuff_hi = shuff(contrast==2);
shuff_lo = shuff(contrast==1);
test.trials = [randsample(shuff_hi, Ntesttrials/2) randsample(shuff_lo, Ntesttrials/2)];
train.trials = setdiff(shuff, test.trials);

train.resp = resp(train.trials,:,:);
test.resp = resp(test.trials,:,:);
train.allsamples = allsamples(:,:,train.trials);
test.allsamples = allsamples(:,:,test.trials);
train.stimval = stimval(train.trials);
test.stimval = stimval(test.trials);

train.contrast = contrast(train.trials);
test.contrast = contrast(test.trials);

%for making figures
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

if p.shrinkmethod == "EM"
    train.bestk = zeros(1,NtimePoints);
end

%summary stats
postmat = zeros(Ntesttrials, p.nbinsstimval, NtimePoints);
uncmat = zeros(NtimePoints,Ntesttrials);
errmat = zeros(NtimePoints,Ntesttrials);
corrmat = zeros(NtimePoints,1);
gammamat = zeros(NtimePoints, p.nboot);

p.nEMiters = 1;

for m = 1:NtimePoints   

    if p.shrinkmethod == "EM"
        if ~exist('forcek', 'var')
        train_samples = squeeze(train.allsamples(m,:,:));
        train_resp = squeeze(train.resp(:,:,1));
        W = (train_resp\train_samples')';
        noise = train_samples - (train_resp*W')';
        sumklikelihood = zeros(1,Nchan-1);
        for i=1:p.nEMiters
            [a, klikelihood, b] = MLE_cov(noise, 1, Nchan-1, 1, 5);
            klikelihoodcell{i,m} = klikelihood;
            sumklikelihood = sumklikelihood+klikelihood;
        end
        testvec = sumklikelihood(2:Nchan-1)-sumklikelihood(1:Nchan-2);
        train.bestk(m) = find(testvec <= 0,1)-1;
%         [bestlike, bestk] = max(klikelihood);
%         train.bestk(m) = bestk;
        else
        train.bestk(m) = forcek;
        end
    end

    [outp, uncertainty, posteriors, gammas] = LDA_decode(m,train,test,p);

    errors = abs(circ_dist(outp, test.stimval));
    avgerrort = mean(errors)/(2*pi);
    avgunct = mean(uncertainty);
    errmat(m,:) = errors;
    uncmat(m,:) = uncertainty;
    fprintf('\n t=%s, error=%3.2f, uncertainty=%3.2f', timestr(m), avgerrort, avgunct);
    postmat(:,:,m) = posteriors;
    corrmat(m) = corr(errors,uncertainty);
    gammamat(m,:) = gammas;
end


% %summary stats
% pooledpost = mean(postmat, 3);
% pooledoutp = zeros(Ntesttrials, 1);
% pooleduncertainty = zeros(Ntesttrials, 1);
% for i = 1:Ntesttrials
%     pooledoutp(i) = mod(circ_mean(p.binvals, pooledpost(i,:)',1), 2*pi);
%     pooleduncertainty(i) = sqrt(sum((circ_dist(p.binvals,pooledoutp(i))).^2 .* pooledpost(i,:)'));
% end
% poolederrors = abs(circ_dist(pooledoutp, stimval(test.trials)));
% avgpoolederror = mean(poolederrors)/(2*pi);
% avgpooledunc = mean(pooleduncertainty);
% 
% %plots you could make
% % 
% col = ["red", "blue"];
% plottrial = 5;
% clear plo
% for m=1:NtimePoints
% pl = plot(p.binvals/2, postmat(plottrial,:,m), 'color', col(1));
% plo(m) = pl(1); hold on
% end
% title('Posteriors at all time points for one test trial');
% xlabel('orientation');
% ylabel('probability');
% xline(stimval(test.trials(plottrial))/2,'--','stimulus orientation');
% legend([plo(1), plo(t0+1)],'before stimulus onset', 'after stimulus onset');
% 
% plot(uncmat);
% xticklabels(timestr(1:end));
% xticks(1:NtimePoints);
% xlabel('time window');
% ylabel('decoded uncertainty');
% % 
% plot(errmat);
% xticklabels(timestr(1:end));
% xticks(1:NtimePoints);
% xlabel('time window');
% ylabel('percent decoding error');
% yline(0.25, '--', 'chance');
% 
% plottrial = 3;
% plot(p.binvals/2,postmat(plottrial,:,9));
% xlabel('orientation');
% ylabel('probability');
% xline(stimval(test.trials(plottrial))/2,'--','stimulus orientation');

