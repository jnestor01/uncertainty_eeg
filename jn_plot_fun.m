function jn_plot_fun(results, partition, opts)

addpath(genpath('/Users/jeffreynestor/Desktop/cupcake/cupcake_data_EEG'));


if ~isfield(opts,'plt')
    opts.plt.x0=10;
    opts.plt.y0 = 10;
    opts.plt.width = 1000;
    opts.plt.height = 600;
    opts.plt.cialpha = 0.1;
    opts.plt.titlesuffix = '';
    opts.plt.linewidth = 0.5;
end

nIters = size(results.params,2);
%     pm = cell2mat(results.params);

if ~iscell(results.testset)
    results = rmfield(results, 'testset');
    results = rmfield(results, 'trainset');
    for i = 1:nIters
        load(results.params{i}.eeg_file);
        ntrials = size(outp.data, 3);
        excludetrials = 1:ntrials;
        excludetrials = ismember(excludetrials, outp.excludetrials);
        ntrials = ntrials - sum(excludetrials);
        nremain = cumsum(~excludetrials);
        trainset = zeros(1, ntrials);
        trialvec = 1:ntrials;
        for j = 1:nremain(end)
            trainset(j) = find(nremain==trialvec(j), 1,'first');
        end
        results.trainset{i} = trainset;
        results.testset{i} = trainset;

    end
end

if isfield(opts, 'iterselect')
    val = string(opts.iterselect.value);
%     if isstring(val)
        iters = eval(sprintf('strcmp(string([pm(:).%s]),%s',opts.iterselect.field,val))';
%     else
%         iters = eval(sprintf('[pm(:).%s]==val',opts.iterselect.field))';
%     end
    iterids = 1:nIters;
    iterids = iterids(iters);
else
    iters = true(nIters,1);
    iterids = 1:nIters;
end

% if the conditions have different time scales you have to change this
NtimePoints = size(results.corrmat{1},1);
t0 = ceil(-results.params{1}.timewindow(1)/results.params{1}.msperbin);

switch partition.type
    case 'lines'
        legend_labels = strings(size(partition.lines,2),1);
        for i = 1:size(partition.lines,2)
            for j = 1:size(partition.cond,2)
                if j==1
                    legend_labels(i) = sprintf( '%s = %s', partition.cond{j}, partition.lines{i}{j} );
                else
                    legend_labels(i) = sprintf( '%s, %s = %s', legend_labels(i), partition.cond{j}, partition.lines{i}{j});
                end
            end
        end
        for b = 1:nIters
            [trialvec{b,1:size(partition.lines,2)}] = deal(true(size(results.testset{b})));
        end
        [itervec{1:size(partition.lines,2)}] = deal(true(1,nIters));
        for a=1:size(partition.cond,2)
            if strcmp(partition.cond{a}, 'contrast')||strcmp(partition.cond{a},'spatialLocation')
                for b=1:nIters
                    if b~=1
                        if ~strcmp(results.params{b-1}.trial_file, results.params{b}.trial_file)
                            trials = load(results.params{b}.trial_file);
                        end
                    else
                        trials = load(results.params{b}.trial_file);
                    end
                    %             oris = trials.expt.p.gratingOrientations;
                    %             orientation = oris(trials.expt.trials(:,1))';
       
                    %             orientation =
                    for c = 1:size(partition.lines,2)
                    if strcmp(partition.cond{a}, 'contrast')
                        contrval = trials.expt.p.gratingContrasts;
                        contrast = string(contrval(trials.expt.trials(results.testset{b},3)));
                        trialvec{b,c} = trialvec{b,c}&strcmp(contrast,partition.lines{c}{a});
                    elseif strcmp(partition.cond{a}, 'spatialLocation')
                        spatialLocation = trials.expt.trials(results.testset{b},4);
                        trialvec{b,c} = trialvec{b,c}&ismember(spatialLocation,partition.lines{c}{a});
                    end

                    end

                        %             condid = find(strcmp(partition.cond{a},trials.expt.trials_headers));
            
                end

            else
                
            for c = 1:size(partition.lines,2)
                    itervec{c} = itervec{c}&eval(sprintf('strcmp(string([pm(:).%s]),partition.lines{c}{a})',partition.cond{a}));
            end
            
            end
        end
        for i=1:size(itervec,2)
            lineiters{i} = 1:nIters;
            lineiters{i} = lineiters{i}(itervec{i});
        end


        if opts.type.errflag
            plots = [];
            figure
            for i = 1:size(partition.lines,2)
                errs = zeros(NtimePoints, sum(itervec{i}));
                for j=1:sum(itervec{i})
                    errs(:,j) = mean(results.errmat{lineiters{i}(j)}(:,trialvec{j,i}),2)/(2*pi);
                end
                meanErr{i} = mean(errs,2);
                plots = [plots plot(meanErr{i}, 'LineWidth', opts.plt.linewidth)]; hold on
               
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = errs;
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end 
                if opts.significance
                    temp = errs;
                    pvals = 1-(sum(temp<0.25,2)/size(temp,2));
                    sig = find(pvals<=opts.statalpha);
                    asts = scatter(sig, meanErr{i}(sig), [], get(plots(end),'Color'), "*");
                end

            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('percent decoding error');
            yline(0.25, '--', 'chance');
            xline(t0, '--', 'stimulus onset');
            pltleg = legend(plots, legend_labels, 'Location', 'northeast', 'Interpreter', 'none');
            title(sprintf('decoding accuracy%s', opts.plt.titlesuffix), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
        end

        if opts.type.predflag
            if ~isfield(opts, 'pred')
                opts.pred = struct();
            end
            if ~isfield(opts.pred, 'timepoint')
                em = cell2mat(results.errmat);
                [mn, mnid] = min(mean(em,2));
                mn
                opts.pred.timepoint =mnid;
            end

            for i = 1:size(partition.lines,2)
                posts = [];
                stimval = [];
                em = [];
                
                for j=1:sum(itervec{i})
                    posts = [posts; results.postmat{lineiters{i}(j)}(trialvec{j,i}, :, opts.pred.timepoint)];
                    em = [em squeeze(results.errmat{lineiters{i}(j)}(opts.pred.timepoint, trialvec{j,i}))];
                    if j~=1
                        if ~strcmp(results.params{lineiters{i}(j)}.trial_file,  results.params{lineiters{i}(j-1)}.trial_file )
                            trials = load(results.params(lineiters{i}(j)));
                        end
                    else
                        trials = load(results.params{lineiters{i}(j)}.trial_file);
                    end
                       
                    spatialLocation = trials.expt.trials(results.testset{lineiters{i}(j)}(trialvec{j,i}),4);
                    stimval = [stimval; spatialLocation];

                end
                mean(em)
                
                opts.pred.title = sprintf('classifier predictions vs presented stimuli at %s, %s%s', results.timestr{1}(opts.pred.timepoint), legend_labels(i), opts.plt.titlesuffix);

%                 size(posts)
%                 size(stimval)
                plot_preds(posts, stimval, opts.pred);


            end
        end


if opts.type.uncflag
            plots = [];
            figure
            for i = 1:size(partition.lines,2)
                unc = zeros(NtimePoints, sum(itervec{i}));
                for j=1:sum(itervec{i})
                    if opts.norm_unc
                        unc(:,j) = normalize(mean(results.uncmat{lineiters{i}(j)}(:,trialvec{j,i}),2));
                    else
                        unc(:,j) = mean(results.uncmat{lineiters{i}(j)}(:,trialvec{j,i}),2);
                    end
                end
                meanUnc{i} = mean(unc,2);
                plots = [plots plot(meanUnc{i}, 'LineWidth', opts.plt.linewidth)]; hold on
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp =unc;
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end
            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('decoded uncertainty');
            xline(t0, '--', 'stimulus onset');
            pltleg = legend(plots, legend_labels, 'Location', 'northeast', 'Interpreter', 'none');
            if opts.norm_unc
            title(sprintf('normalized decoded uncertainty%s', opts.plt.titlesuffix), 'Interpreter', 'none');
            else
            title(sprintf('decoded uncertainty%s', opts.plt.titlesuffix), 'Interpreter', 'none');
            end
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end



            if opts.type.corrflag
            plots = [];
            figure
            for i = 1:size(partition.lines,2)
                corrs = zeros(NtimePoints, sum(itervec{i}));
                for j=1:sum(itervec{i})
%                     corrs(:,j) = diag(corr(results.uncmat{lineiters{i}(j)}(:,trialvec{j,i})', results.errmat{lineiters{i}(j)}(:,trialvec{j,i})'));
                    for k = 1:NtimePoints
                        corrs(k,j) = corr(results.uncmat{lineiters{i}(j)}(k,trialvec{j,i})', results.errmat{lineiters{i}(j)}(k,trialvec{j,i})');
                    end
                end
                meanCorr{i} = mean(corrs,2);
                plots = [plots plot(meanCorr{i}, 'LineWidth', opts.plt.linewidth)]; hold on
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = corrs;
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end
                if opts.significance
                    temp = corrs;
                    pvals = 1-(sum(temp>0,2)/size(temp,2));
                    sig = find(pvals<=opts.statalpha);
                    asts = scatter(sig, meanCorr{i}(sig), [], get(plots(end),'Color'), "*");
                end
            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('pearson r');
            xline(t0, '--', 'stimulus onset');
            pltleg = legend(plots, legend_labels, 'Location', 'northeast', 'Interpreter', 'none');
            title(sprintf('correlation between decoding error and decoded uncertainty%s', opts.plt.titlesuffix), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end

            if opts.type.predflag
                
            end

    case 'cond'
        condStr = strings(1,nIters);
        for i = 1:nIters
            condStr(i) = results.params{i}.(partition.cond);
        end
        [conds, cond_a, cond_c] = unique(condStr);


        for plot_conds_ind = 1:size(opts.plot_conds_array,1)
            plot_conds = opts.plot_conds_array(plot_conds_ind,:);
            plots = [];

            if opts.type.errflag
            figure
            for i = plot_conds
                condTrials = 1:nIters;
                condTrials = condTrials((cond_c==i)&iters);
                errs = zeros(size(results.errmat{condTrials(1)},1),size(results.errmat{condTrials(1)},2),sum(cond_c==i&iters));
                for j=1:sum(cond_c==i&iters)
                    errs(:,:,j) = results.errmat{condTrials(j)}/(2*pi);
                end
                meanErr{i} = mean(errs,[2,3]);
                plots = [plots plot(meanErr{i}, 'LineWidth', opts.plt.linewidth)]; hold on
               
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = squeeze(mean(errs,2));
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end 
                if opts.significance
                    temp = squeeze(mean(errs,2));
                    pvals = 1-(sum(temp<0.25,2)/size(temp,2));
                    sig = find(pvals<=opts.statalpha);
                    asts = scatter(sig, meanErr{i}(sig), [], get(plots(end),'Color'), "*");
                end

            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('percent decoding error');
            yline(0.25, '--', 'chance');
            xline(t0, '--', 'stimulus onset');
            pltleg = legend(plots, conds(plot_conds), 'Location', 'northeast');
            title(sprintf('decoding accuracy by %s', partition.cond), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end

            if opts.type.uncflag
            plots = [];
            figure
            for i = plot_conds
                condTrials = 1:nIters;
                condTrials = condTrials(cond_c==i&iters);
                unc = zeros(size(results.uncmat{condTrials(1)},1),size(results.uncmat{condTrials(1)},2),sum(cond_c==i&iters));
                for j=1:sum(cond_c==i&iters)
                    if opts.norm_unc
                        unc(:,:,j) = normalize(results.uncmat{condTrials(j)});
                    else
                        unc(:,:,j) = results.uncmat{condTrials(j)};
                    end
                end
                meanUnc{i} = mean(unc,[2,3]);
                plots = [plots plot(meanUnc{i}, 'LineWidth', opts.plt.linewidth)]; hold on
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = squeeze(mean(unc,2));
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end
            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('decoded uncertainty');
            xline(t0, '--', 'stimulus onset');
            legend(plots, conds(plot_conds));
            title(sprintf('decoded uncertainty by %s', partition.cond), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end


            if opts.type.corrflag
            plots = [];
            figure
            for i = plot_conds
                condTrials = 1:nIters;
                condTrials = condTrials(cond_c==i&iters);
                corrs = zeros(size(results.corrmat{condTrials(1)},1),sum(cond_c==i&iters));
                for j=1:sum(cond_c==i&iters)
                    corrs(:,j) = results.corrmat{condTrials(j)};
                end
                meanCorr{i} = mean(corrs,2);
                plots = [plots plot(meanCorr{i}, 'LineWidth', opts.plt.linewidth)]; hold on
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = corrs;
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end
                if opts.significance
                    temp = corrs;
                    pvals = 1-(sum(temp>0,2)/size(temp,2));
                    sig = find(pvals<=opts.statalpha);
                    asts = scatter(sig, meanCorr{i}(sig), [], get(plots(end),'Color'), "*");
                end
            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('pearson r');
            xline(t0, '--', 'stimulus onset');
            legend(plots, conds(plot_conds));
            title(sprintf('correlation between decoding error and decoded uncertainty by %s', partition.cond), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end


            if opts.type.gammaflag
            plots = [];
            figure
            for i = plot_conds
                condTrials = 1:nIters;
                condTrials = condTrials(cond_c==i&iters);
                gammas = zeros(size(results.gammamat{condTrials(1)},1),size(results.gammamat{condTrials(1)},2),sum(cond_c==i&iters));
                for j=1:sum(cond_c==i&iters)
                    gammas(:,:,j) = results.gammamat{condTrials(j)};
                end
                meanGamma{i} = mean(gammas,[2,3]);
                plots = [plots plot(meanGamma{i}, 'LineWidth', opts.plt.linewidth)]; hold on
            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('average computed shrinkage parameter (gamma)');
            xline(t0, '--', 'stimulus onset');
            legend(plots, conds(plot_conds));
            title(sprintf('shrinkage dynamics by %s', partition.cond), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end

            if opts.type.predflag
            if ~isfield(opts, 'pred')
                opts.pred = struct();
            end
            if ~isfield(opts.pred, 'timepoint')
                em = cell2mat(results.errmat);
                [mn, mnid] = min(mean(em,2));
                opts.pred.timepoint =mnid;
            end

            for i = plot_conds
                condTrials = 1:nIters;
                condTrials = condTrials(cond_c==i&iters);
                posts = [];
                stimval = [];
                for j=1:sum(cond_c==i&iters)
                    posts = [posts; results.postmat{condTrials(j)}(:, :, opts.pred.timepoint)];
                    if j~=1
                        if ~strcmp(results.params{condTrials(j)}.trial_file,  results.params{condTrials(j-1)}.trial_file )
                            trials = load(results.params{condTrials(j)}.trial_file);
                        end
                    else
                        trials = load(results.params{condTrials(j)}.trial_file);
                    end
                       
                    spatialLocation = trials.expt.trialsPresented.theta(results.testset{condTrials(j)});
                    stimval = [stimval; spatialLocation];                  
                end           
                opts.pred.title = sprintf('classifier predictions vs presented stimuli at %s, %s%s', results.timestr{1}(opts.pred.timepoint), conds(i), opts.plt.titlesuffix);
                plot_preds(posts, stimval', opts.pred);
            end
        end



        end

    case 'stim'
        trials = load(results.params{1}.trial_file);
        featureid = find(strcmp(trials.expt.trials_headers, partition.cond));

        feature_cond = trials.expt.trials(:,featureid);
        [conds, cond_a, cond_c] = unique(feature_cond);

        paramname = sprintf('%ss',partition.cond);


        condStr = strings(numel(conds),1);
        for k = 1:numel(conds)
            condStr(k) = sprintf('%s = %g', partition.cond, trials.expt.p.(paramname)(conds(k)));
        end

            plot_conds = 1:numel(conds);
            plots = [];

            if opts.type.errflag
            figure
            for i = plot_conds
                errs = [];
                for j=1:nIters
                    errs = cat(2, errs, mean(results.errmat{j}(:,feature_cond(results.testset{j})==conds(i)),2)/(2*pi));
                end
                meanErr{i} = mean(errs,2);
                plots = [plots plot(meanErr{i}, 'LineWidth', opts.plt.linewidth)]; hold on
               
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = errs;
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end 
                if opts.significance
                    temp = errs;
                    pvals = 1-(sum(temp<0.25,2)/size(temp,2));
                    sig = find(pvals<=opts.statalpha);
                    asts = scatter(sig, meanErr{i}(sig), [], get(plots(end),'Color'), "*");
                end

            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('percent decoding error');
            yline(0.25, '--', 'chance');
            xline(t0, '--', 'stimulus onset');
            legend(plots, condStr(plot_conds));
            title(sprintf('decoding accuracy by %s', partition.cond), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end

            if opts.type.uncflag
            plots = [];
            figure
            for i = plot_conds
                unc = [];
                for j=1:nIters
                    unc = cat(2, unc, mean(results.uncmat{j}(:,feature_cond(results.testset{j})==conds(i)),2)/(2*pi));
                end
                if opts.norm_unc
                    unc = normalize(unc,1);
                end
                meanUnc{i} = mean(unc,2);
                plots = [plots plot(meanUnc{i}, 'LineWidth', opts.plt.linewidth)]; hold on
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = unc;
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end
            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('decoded uncertainty');
            xline(t0, '--', 'stimulus onset');
            legend(plots, condStr(plot_conds));
            title(sprintf('decoded uncertainty by %s', partition.cond), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end


            if opts.type.corrflag
            plots = [];
            figure
            for i = plot_conds
                corrs = [];
                for h=1:sum(iters)
                    j = 1:nIters;
                    j = j(iters(h));
                    corr_j = corr( results.errmat{j}(:,feature_cond(results.testset{j})==conds(i))', results.uncmat{j}(:,feature_cond(results.testset{j})==conds(i))' );
                    corrs = cat(2, corrs, corr_j);
                end
                meanCorr{i} = mean(corrs,2);
                plots = [plots plot(meanCorr{i}, 'LineWidth', opts.plt.linewidth)]; hold on
                if opts.ci
                    %"Bootstrap" across iterations or across test trials
                    %or both? Across iterations for now.
                    temp = corrs;
                    temp = sort(temp,2);
                    rown = ceil(size(temp,2)*opts.statalpha);
                    upper = temp(:,size(temp,2)-rown)';
                    lower = temp(:,rown)';
                    xs = 1:NtimePoints;
                    patch([xs fliplr(xs)], [lower fliplr(upper)], get(plots(end),'Color'), 'FaceAlpha', opts.plt.cialpha, 'EdgeColor', 'none');
                end
                if opts.significance
                    temp = corrs;
                    pvals = 1-(sum(temp>0,2)/size(temp,2));
                    sig = find(pvals<=opts.statalpha);
                    asts = scatter(sig, meanCorr{i}(sig), [], get(plots(end),'Color'), "*");
                end
            end
            xticklabels(results.timestr{1}(1:2:NtimePoints));
            xticks(1:2:NtimePoints);
            xlabel('time window');
            ylabel('pearson r');
            xline(t0, '--', 'stimulus onset');
            legend(plots, condStr(plot_conds));
            title(sprintf('correlation between decoding error and decoded uncertainty by %s', partition.cond), 'Interpreter', 'none');
            set(gcf, 'position', [opts.plt.x0 opts.plt.y0 opts.plt.width opts.plt.height]);
            end
        

end

end