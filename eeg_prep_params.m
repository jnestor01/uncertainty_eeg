function p=eeg_prep_params

p.filters = [0.01, 0];
p.suffix = 'aref_hfilt.mat';
p.rm_baseline_window = [-500:0];
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
end