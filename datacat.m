function datacat(trialsfiles, eegfiles, dest, trialsfilename, eegfilename)
%trials a cell containing names of trial files in order
% eeg a cell containing names of eeg files in order
% dest name of the folder in path at the top of this script to save the
% output in
% filename name of the output
datapath = '/Users/jeffreynestor/Desktop/Uncertainty/EEG_data';
addpath(genpath(datapath));

N = numel(trialsfiles);

load(trialsfiles{1});
trialsfields = fieldnames(expt.trialsPresented);

expt.trialsPresented.session  = ones(size(expt.trialsPresented.theta));

load(eegfiles{1});

for i = 2:N
    new = load(trialsfiles{i});
    for j = 1:numel(trialsfields)
        expt.trialsPresented.(trialsfields{j}) = [expt.trialsPresented.(trialsfields{j}), new.expt.trialsPresented.(trialsfields{j})];
    end
    expt.trialsPresented.session = [expt.trialsPresented.session, (i*ones(size(new.expt.trialsPresented.theta)))];
    new = load(eegfiles{i});
    outp.excludetrials = [outp.excludetrials, new.outp.excludetrials+size(outp.data,3)];
    outp.data = cat(3, outp.data, new.outp.data);
    if exist('outp.interpchans','var')&&exist('new.outp.interpchans','var')
    outp.interpchans = unique([outp.interpchans, new.outp.interpchans]);
    elseif exist('new.outp.interpchans','var')
    outp.interpchans = new.outp.interpchans;
    elseif ~exist('outp.interpchans', 'var')
        outp.interpchans = [];
    end
    outp.excludechannels = unique([outp.excludechannels, new.outp.excludechannels]);
end

cd(datapath);
if isfolder(dest)
    cd(dest);
else
    mkdir(dest);
    cd(dest);
end

save(trialsfilename, 'expt', '-v7.3');
save(eegfilename, 'outp', '-v7.3');
end