% EEGLAB history file generated on the 29-Feb-2024
% ------------------------------------------------
% not set up to downsample, add if desired
filters = [0.01,0];
suffix = 'aref_hfilt.mat';
rm_baseline_window = [-500:0];
eeg_chans = [1:19 21:63];
diode_thresh = 1*10^4;
epoch_after_delay_correction = [-1000 1500];

p.eyeparams.do = 1;
p.eyeparams.center = [960, 540];
p.eyeparams.blinkwindow = [0 500]-epoch_after_delay_correction(1);
p.eyeparams.saccwindow = [0 0];
p.eyeparams.gazewindow = p.eyeparams.blinkwindow;
fileloc = '/Users/jeffreynestor/Desktop/Uncertainty/EEG_data/Pilot_6_report';
filename = 'cupcake0006';

hz = 1000;

% fileloc = '/Users/jeffreynestor/Desktop/Uncertainty/EEG_data/Pilot_1/';
% filename = 'cupcake0001';

% fileloc = '/Users/jeffreynestor/Desktop/Uncertainty/EEG_data/Pilot_2/cupcake0002_small';
% filename = 'cupcake0002_small';
% % 
% fileloc = '/Users/jeffreynestor/Desktop/Uncertainty/EEG_data/Pilot_2/cupcake0002_large';
% filename = 'cupcake0002_large';
addpath(fileloc);

[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;


EEG.etc.eeglabvers = '2023.1'; % this tracks which version of EEGLAB is being used, you may ignore it
EEG = pop_loadbv(fileloc, sprintf('%s.vhdr', filename));

EEG = pop_basicfilter(EEG,1:63,'Cutoff',filters,'Design','butter','Filter','bandpass','Order',2);
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'setname', 'filtered', 'gui', 'off');

% rereference
EEG = pop_eegchanoperator(EEG, { ...
    'ch1=ch1-.5*ch20 label Fp1' ...
    'ch2=ch2-.5*ch20 label Fz' ...
    'ch3=ch3-.5*ch20 label F3' ...
    'ch4=ch4-.5*ch20 label F7' ...
    'ch5=ch5-.5*ch20 label FT9' ...
    'ch6=ch6-.5*ch20 label FC5' ...
    'ch7=ch7-.5*ch20 label FC1' ...
    'ch8=ch8-.5*ch20 label C3' ...
    'ch9=ch9-.5*ch20 label T7' ...
    'ch10=ch10-.5*ch20 label CP5' ...
    'ch11=ch11-.5*ch20 label CP1' ...
    'ch12=ch12-.5*ch20 label Pz' ...
    'ch13=ch13-.5*ch20 label P3' ...
    'ch14=ch14-.5*ch20 label P7' ...
    'ch15=ch15-.5*ch20 label O1' ...
    'ch16=ch16-.5*ch20 label Oz' ...
    'ch17=ch17-.5*ch20 label O2' ...
    'ch18=ch18-.5*ch20 label P4' ...
    'ch19=ch19-.5*ch20 label P8' ...
    'ch20=ch20 label TP10' ...
    'ch21=ch21-.5*ch20 label CP6' ...
    'ch22=ch22-.5*ch20 label CP2' ...
    'ch23=ch23-.5*ch20 label Cz' ...
    'ch24=ch24-.5*ch20 label C4' ...
    'ch25=ch25-.5*ch20 label T8' ...
    'ch26=ch26-.5*ch20 label FT10' ...
    'ch27=ch27-.5*ch20 label FC6' ...
    'ch28=ch28-.5*ch20 label FC2' ...
    'ch29=ch29-.5*ch20 label F4' ...
    'ch30=ch30-.5*ch20 label F8' ...
    'ch31=ch31-.5*ch20 label Fp2' ...
    'ch32=ch32-.5*ch20 label AF7' ...
    'ch33=ch33-.5*ch20 label AF3' ...
    'ch34=ch34-.5*ch20 label AFz' ...
    'ch35=ch35-.5*ch20 label F1' ...
    'ch36=ch36-.5*ch20 label F5' ...
    'ch37=ch37-.5*ch20 label FT7' ...
    'ch38=ch38-.5*ch20 label FC3' ...
    'ch39=ch39-.5*ch20 label C1' ...
    'ch40=ch40-.5*ch20 label C5' ...
    'ch41=ch41-.5*ch20 label TP7' ...
    'ch42=ch42-.5*ch20 label CP3' ...
    'ch43=ch43-.5*ch20 label P1' ...
    'ch44=ch44-.5*ch20 label P5' ...
    'ch45=ch45-.5*ch20 label PO7' ...
    'ch46=ch46-.5*ch20 label PO3' ...
    'ch47=ch47-.5*ch20 label POz' ...
    'ch48=ch48-.5*ch20 label PO4' ...
    'ch49=ch49-.5*ch20 label PO8' ...
    'ch50=ch50-.5*ch20 label P6' ...
    'ch51=ch51-.5*ch20 label P2' ...
    'ch52=ch52-.5*ch20 label CPz' ...
    'ch53=ch53-.5*ch20 label CP4' ...
    'ch54=ch54-.5*ch20 label TP8' ...
    'ch55=ch55-.5*ch20 label C6' ...
    'ch56=ch56-.5*ch20 label C2' ...
    'ch57=ch57-.5*ch20 label FC4' ...
    'ch58=ch58-.5*ch20 label FT8' ...
    'ch59=ch59-.5*ch20 label F6' ...
    'ch60=ch60-.5*ch20 label AF8' ...
    'ch61=ch61-.5*ch20 label AF4' ...
    'ch62=ch62-.5*ch20 label F2' ...
    'ch63=ch63-.5*ch20 label Iz' ...
    });
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'setname', 'reref', 'gui', 'off');

% 
% EEG = pop_runica(EEG,'extended',1,'chanind',eeg_chans);
% [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'setname', 'ica', 'gui', 'off');
% 
% pop_topoplot(EEG,0,1:30,'',[5 6],0,'electrodes','off');
% pop_eegplot(EEG,0,1,1);

% pop_eegplot(EEG)

EEG = pop_epoch( EEG, {  'S  8'  }, [-1  2]);
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'setname', 'epochs', 'gui', 'off');

shft_eeg = shiftdim(EEG.data,1);
dtrn_eeg = shiftdim(detrend(shft_eeg),2);

a = epoch_after_delay_correction(1);
b = epoch_after_delay_correction(2);

zero = find(EEG.times==0);
first_sample = zero+a;
last_sample = zero+b;

outp.data = zeros(size(dtrn_eeg,1), b-a+1, size(dtrn_eeg,3));
delaystest = (EEG.data(64,[zero:end],:)-min(EEG.data(64,[zero:end],:), [], 2))>diode_thresh;
for i=1:size(delaystest,3)
    delay = find(delaystest(:,:,i),1,'first');
    outp.data(:,:,i) = dtrn_eeg(:, first_sample+delay:last_sample+delay, i);
end
outp.time = EEG.times(first_sample:last_sample);


means = mean(outp.data(:,ismember(outp.time,rm_baseline_window),:),2);
outp.data = single(outp.data - means);

% EEG.data = outp.data;

outp.chanlocs = EEG.chanlocs;
outp.hz = hz;

absmaxes = max(abs(outp.data(eeg_chans,:,:)),[],[2]);
channelmeans = mean(absmaxes,3);

for i=1:64
    EEG.chanlocs(i).urchan = i;
end
excludechans = 1:64;
excludechans(eeg_chans) = [];

figure
histogram(channelmeans);
title('mean max absolute amplitude across trials at each channel to judge outliers')
nbadchan = input('how many channels to interpolate?');
if nbadchan>0
    EEG_copy = EEG;
    EEG_copy.data = outp.data;
    [~,worst] = sort(channelmeans, 'descend');
    outp.interpchans = eeg_chans(worst(1:nbadchan));
    eeg_chans2 = eeg_chans(worst(nbadchan+1:end));
    EEG_copy = erplab_interpolateElectrodes(EEG_copy, outp.interpchans, excludechans);
    outp.data = EEG_copy.data;
else
    eeg_chans2 = eeg_chans;
end

absmaxes = max(abs(outp.data(eeg_chans2,:,:)),[],[2]);

trialmaxes = squeeze(max(absmaxes,[],1));
figure
histogram(trialmaxes);
title('maximum absolute amplitude across remaining channels at each trial to judge outliers');
trialthresh = input('set an upper threshold for exclusion');
exclude_trials = find(trialmaxes>trialthresh);

if p.eyeparams.do
    eyefile = sprintf('%s_eyefile.edf', filename);
    eyedata = Edf2Mat(eyefile);
    eye_rejects = eyereject(eyedata, p.eyeparams);
    exclude_trials = [exclude_trials', eye_rejects];
    exclude_trials = unique(exclude_trials);
end

outp.excludetrials = exclude_trials;
outp.excludechannels = excludechans;
cd(fileloc);
save(sprintf('%s_%s', filename, suffix), 'outp');


%%

t0 = -epoch_after_delay_correction(1);
trials = load(sprintf('%s.mat', filename));
stimval = trials.expt.trials(:,4);
fixes = trials.expt.trials(:,8);
stimval(outp.excludetrials) = [];
fixes(outp.excludetrials) = [];
left = fixes==1&stimval>90&stimval<270;
right = ~left&stimval~=90&stimval~=270&fixes==1;

plotchannels = eeg_chans;

binsize =25;
firstlat = 0;
lastlat = 475;
latency_edges = [[firstlat:binsize:lastlat]' [firstlat:binsize:lastlat]'+binsize];

latency_bins = t0+repmat(0:(binsize-1), numel(latency_edges(:,1)),1) + repmat(latency_edges(:,1),1,binsize);
plotdata = outp.data(plotchannels,:,:);
plotdata(:,:,outp.excludetrials) = [];
upper = max(plotdata, [],'all');
lower = min(plotdata, [],'all');

upper = 4;
lower = -4;

figure
for i = 1:size(latency_bins,1)
    left_erp{i} = mean(plotdata(:,latency_bins(i,:),left), [2,3]);
    right_erp{i} = mean(plotdata(:,latency_bins(i,:),right), [2,3]);
%     pl_r{i} = nexttile;
    subplot(size(latency_bins,1),2,(2*(i-1))+1);
    topoplot(left_erp{i}, outp.chanlocs(plotchannels), 'maplimits',[lower,upper], 'conv','on', 'intsquare', 'off');
    title(sprintf('ERPs for stimuli in the left hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)));

%     title(pl_r{i}, 'txt',sprintf('ERPs for stimuli in the left hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)));
    
%     pl_l{i} = nexttile;
% nexttile
    subplot(size(latency_bins,1),2,(2*(i-1))+2);
 
    topoplot(right_erp{i}, outp.chanlocs(plotchannels), 'maplimits',[lower,upper], 'conv','on', 'intsquare', 'off');
    title(sprintf('ERPs for stimuli in the right hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)));
    cbar(0,0,[lower, upper]);
    
%     title(pl_l{i}, sprintf('ERPs for stimuli in the left hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)));
%     title(sprintf('ERPs for stimuli in the left hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)), 'target', pl_l{i});

end

upper = 2;
lower = -2;

%difference in pattern z-score
figure
for i = 1:size(latency_bins,1)
    left_erp{i} = mean(plotdata(:,latency_bins(i,:),left), [2,3]);
    right_erp{i} = mean(plotdata(:,latency_bins(i,:),right), [2,3]);
    left_erp{i} = (left_erp{i} - mean(left_erp{i}))/std(left_erp{i});
    right_erp{i} = (right_erp{i} - mean(right_erp{i}))/std(right_erp{i});
    
%     pl_r{i} = nexttile;
%     subplot(size(latency_bins,1),1,i);
    subplot(size(latency_bins,1)/4,4,i);

    topoplot(right_erp{i}-left_erp{i}, outp.chanlocs(plotchannels), 'maplimits',[lower,upper], 'conv','on', 'intsquare', 'off');

    title(sprintf('%ims-%ims', latency_edges(i,1), latency_edges(i,2)));
%     title(pl_r{i}, 'txt',sprintf('ERPs for stimuli in the left hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)));
    
%     pl_l{i} = nexttile;
% nexttile
    cbar(0,0,[lower, upper]);
    
%     title(pl_l{i}, sprintf('ERPs for stimuli in the left hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)));
%     title(sprintf('ERPs for stimuli in the left hemifield, %ims-%ims', latency_edges(i,1), latency_edges(i,2)), 'target', pl_l{i});

end
suptit = sgtitle('difference in pattern z-score for ERPs to stimuli in the right hemifield minus left hemifield');
set(suptit, 'Pargin', 5);


timecourse_left = mean(outp.data(:,:,left),3);
timecourse_right = mean(outp.data(:,:,right),3);
% plottime = [-100:300]+t0;
plottime = [500:2000];
targetchannels_array = [1:16; 17:32; 33:48];

for k = 1:size(targetchannels_array,1)
    targetchannels = targetchannels_array(k,:);
        figure
        tiledlayout(8,2)    
for i=targetchannels
    nexttile
    plot(timecourse_left(i,plottime)); hold on;
    plot(timecourse_right(i,plottime));
    title(sprintf('channel no %i : %s', i, outp.chanlocs(i).labels))
    ylim([-5 15])
end
end
% 
% for k = 1:size(targetchannels_array,1)
%     targetchannels = targetchannels_array(k,:);
%         figure
%         tiledlayout(8,2)    
% for i=targetchannels
%     nexttile
% for j = 1:2160
% plot(outp.data(23,:,j)); hold on
% end
%     title(sprintf('channel no %i : %s', i, outp.chanlocs(i).labels))
% end
% end

targetchannels_array = [1:16; 17:32; 33:48; 49:62];

for k = 1:size(targetchannels_array,1)
    targetchannels = targetchannels_array(k,:);
        figure
        tiledlayout(8,2)    
for i=targetchannels
    nexttile
plot(mean(plotdata(i,:,:),3))
end
end