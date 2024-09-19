function p=eeg_prep_params

p.filters = [0.01, 0];
p.suffix = 'aref_hfilt.mat';
p.rm_baseline = 0;
p.rm_baseline_window = [-200:0];
p.eeg_chans = [1:19 21:63];
p.diode_thresh = 1*10^4;
p.epoch_after_delay_correction = [-1000 1500];
p.eyeparams.do = 1;
p.eyeparams.center = [960, 540];
p.eyeparams.blinkwindow = [0 500]-epoch_after_delay_correction(1);
p.eyeparams.saccwindow = [0 0];
p.eyeparams.gazewindow = p.eyeparams.blinkwindow;
p.fileloc = '/projectnb/rdenlab/Users/Data/Cupcake_EEG';
p.subjfolder = '/S0108';
p.filename = '/cupcake_jeff_0916';
p.hz = 1000;
p.reference = 'avg';

%Plot flags
%topoplots
p.plots.fullavg = 1;
p.plots.cueduncued = 1;
p.plots.leftright = 1;
%erps
p.plots.channelavg = 1;
p.plots.channelavg_cueduncued = 1;
p.plots.channelavg_leftright = 1;
%plot timeseries of specific channels?
p.plots.somechannels = {'Oz'};

%for topoplots
p.plots.dims_topo = [600, 1200];
p.plots.binsize =25;
p.plots.firstlat = 0;
p.plots.lastlat = 475;
%colorbar scale, in mV
p.plots.upperbound = 4;
p.plots.lowerbound = -4;

%for timeseries
p.plots.dims_timeseries = [1000, 600];
p.plots.epoch = [-500, 750];
p.plots.tickinterval = 100;

p.plots.format = 'png';
end