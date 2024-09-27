datafolder = '/Users/jeffreynestor/Desktop/Uncertainty/Batch_Results/Unprocessed';
addpath(genpath('/Users/jeffreynestor/Desktop/Uncertainty'));

datasets = dir(datafolder);
ignore = [];
for i = 1:numel(datasets)
    if strcmp(datasets(i).name, '.')||strcmp(datasets(i).name, '..')
        ignore = [ignore i];
    end
end
datasets(ignore) = [];
for i = 1:numel(datasets)
    datafile = timecat(sprintf('%s/%s/results.mat', datafolder, datasets(i).name));
    allresults(i) = load(datafile);
end

sortparams = {'trial_file', 'covmethod', 'regressionmethod', 'loocv'};
for i = 1:numel(sortparams)
    sortparamids{i} = [];
    paramlists{i} = {};
end
for i = 1:numel(allresults)
    for j  = 1:numel(sortparams)
        paramlists{j} = union(paramlists{j}, string(allresults(i).results.params{1}.(sortparams{j})));
        sortparamids{j} = [sortparamids{j}, find(strcmp(string(allresults(i).results.params{1}.(sortparams{j})), paramlists{j}))];
    end
end

ntimepoints = 60;
loocv = sortparamids{4}==find(paramlists{4}=='1');
timestr = allresults(find(~loocv, 1, 'first')).results.timestr{1};


%grand average error
for i = 1:numel(paramlists{2})
    ids = find(loocv&(sortparamids{2}==i));
    subjmeans{i} = nan(ntimepoints, numel(ids));
    for j = 1:numel(ids)
        %subject mean
        figure
        subjmeans{i}(:,j) = mean(allresults(ids(j)).results.errmat{1}, 2, 'omitnan');
        plot(1:ntimepoints, subjmeans{i}(:,j)/(2*pi));
        title(datasets(ids(j)).name, 'Interpreter', 'none');
   
    end

end
xticks(1:5:60)
xticklabels(timestr{1:5:60})