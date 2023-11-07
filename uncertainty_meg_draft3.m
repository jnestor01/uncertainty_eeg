%Kok et al. 2017 LDA decoding method with the differences being the use of bootstrapping and a distribution of
%likelihoods is computed instead of a point estimate

%compared to TAFKAP, this method does not give particular attention to
%tuning-correlated noise

%portion of data to hold out as testing data
p.test_part = 0.2;
%maximum number of bootstrap iterations
p.nboot = 5;
%number of basis functions to use for defining tuning curves
%this parameter must be less than the number of unique stimulus values
%presented across trials
p.nchan = 6;
%number of sets of basis functions to choose from during bootstrapping
p.nsets = 4;
%exponent to which basis functions are raised
p.chan_exp = 4;
%number of bins to discretize possible stimulus values
p.nbinsstimval = 100;
%size of time bins
p.msperbin = 25;
%calculate shrinkage parameter numerically or by cross-validation
p.shrinkmethod = 'numeric';
%p.shrinkmethod = 'cv';

runs = [1:8];
meg = load('R1507_CupcakeAperture_4.25.19_ebci_condData.mat');
timewindow = [1001:2001];
targetchannels = [1:157];
trialsPerRun = size(meg.D.condData,3);

%stim onset, not used in decoding just for making strings
t0 = 1000;

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
Nchan = size(targetchannels,2);
stimval = stimval/180 * 2 * pi;
%time binning
% cut off data at end if trial length is not divisible by binsize
NtimePoints = floor(size(allsamples,1)/p.msperbin);
allsamples = allsamples(1:NtimePoints*p.msperbin,:,:);
allsamples = reshape(mean(reshape(allsamples,p.msperbin,[]),1),NtimePoints,Nchan,Ntrials);
%data are demeaned s.t. the average for each channel is zero
means = mean(allsamples, [1 3]);
allsamples = bsxfun(@minus, allsamples, means);
%%

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
%%

Ntesttrials = floor(p.test_part*Ntrials);
Ntraintrials = Ntrials-Ntesttrials;
shuff = randperm(Ntrials);
test.trials = shuff(1:Ntesttrials)';
train.trials = shuff(Ntesttrials+1:end)';
train.resp = resp(train.trials,:,:);
test.resp = resp(test.trials,:,:);
train.allsamples = allsamples(:,:,train.trials);
test.allsamples = allsamples(:,:,test.trials);

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

%summary stats
postmat = zeros(Ntesttrials, p.nbinsstimval, NtimePoints);
uncmat = zeros(NtimePoints,1);
errmat = zeros(NtimePoints,1);

for m = 1:NtimePoints    
    [outp, uncertainty, posteriors] = LDA_decode(m,train,test,p);

    errors = abs(circ_dist(outp, stimval(test.trials)));
    avgerrort = mean(errors)/(2*pi);
    avgunct = mean(uncertainty);
    errmat(m) = avgerrort;
    uncmat(m) = avgunct;
    fprintf('\n t=%s, error=%3.2f, uncertainty=%3.2f', timestr(m), avgerrort, avgunct);
    postmat(:,:,m) = posteriors;
end

