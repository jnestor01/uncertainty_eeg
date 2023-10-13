%ignoring random seeding for now

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
Ntrials = 100;
Nelectrodes = 20;
stimval = rand(Ntrials,1) * 2*pi;
trialLength = 20;
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
samples = reshape(mean(reshape(samples,p.msperbin,[])),NtimePoints,Nelectrodes,Ntrials);

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

for b=1:p.nboot
%find W and Sigma
    set = randi(p.nsets);

    N = size(train_resp, 1);
    idx = randi(N,N,1);
    
    %calculate matrix and noise
    W = (train_resp(idx,:, set)\train_samples(:,idx)')';
    noise = train_samples(idx,:) - (train_resp(idx,:,set)*W');

    %calculate sigma
    samplecov = noise'*noise/Ntraintrials;

end

