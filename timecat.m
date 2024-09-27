function outfile = timecat(infile)
%concatenates datasets where timepoints are split into different structs
%(loocv)
inp = load(infile);
[pth, nm, ext] = fileparts(infile);
if all(inp.results.params{1}.timewindow==inp.results.params{2}.timewindow)
    outfile = infile;
    error('%s is not in timecattable format', infile);    
    return
end
outfile = sprintf('%s/%s_timecat.mat', pth, nm);

niters = numel(inp.results.params);

windows = zeros(2,niters);
for i = 1:niters
    windows(:,i) = inp.results.params{i}.timewindow;
end
[timepoints, timeids] = sort(windows,2);
if ~isequal(timeids(1,:), timeids(2,:))
    error(1, 'error: can''t align time windows');
    timeids = timeids(1,:);
end
results.errmat{1} = [];
results.uncmat{1} = [];
results.corrmat{1} = [];
results.gammamat{1} = [];
results.timestr{1} = [];
results.postmat{1} = [];

for i = 1:niters
    results.errmat{1} = [results.errmat{1}; inp.results.errmat{i}];
    results.uncmat{1} = [results.uncmat{1}; inp.results.uncmat{i}];
    results.corrmat{1} = [results.corrmat{1}; inp.results.corrmat{i}];
    results.gammamat{1} = [results.gammamat{1}; inp.results.gammamat{i}];
    results.timestr{1} = [results.timestr{1}; inp.results.timestr{i}];
    results.postmat{1} = cat(3, results.postmat{1}, inp.results.postmat{i});
end
results.params{1} = inp.results.params{1};
results.params{1}.timewindow = [min(windows, [], 'all'), max(windows, [], 'all')];

results.testset = inp.results.testset{1};
results.trainset = results.testset;

save(outfile, 'results');
end
% sp = 1;
% %some of mine don't have proper test sets
% if sp
%     eeg = load(results.params{1}.eeg_file);
%     removeTrials = false(size(eeg.outp.data,2),1);
% 
% if isfield(eeg.outp, 'excludetrials')
%     removeTrials(eeg.outp.excludetrials) = true;
% end
% end