addpath(genpath('/Users/jeffreynestor/Desktop/cupcake/cupcake_data_EEG'));
filenames = {'cupcake_S0108_2cat', 'cupcake_S0004_2cat', 'cupcake_S0003_2cat'};
subjdirs = {'S0108', 'S0004', 'S0003'};

ext = 'png';

for i = 1:numel(filenames)
    expts(i) = load(filenames{i});
end

%% Individual plots
for i = 1:numel(filenames)
    filename = filenames{i};
dirname = sprintf('%s/%s_behav_plots', subjdirs{i}, filename);
if ~exist(dirname, 'dir')
    mkdir(dirname);
end
expt = expts(i).expt;

hist_range = [-pi/9, pi/9];
nbins = 36;
hist_edges = linspace(hist_range(1), hist_range(2), nbins);
hist_edges_deg = rad2deg(hist_edges);


error = figure;
histogram(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==0)), hist_edges_deg(2:end), 'Normalization', 'probability', 'EdgeAlpha', 0, 'FaceAlpha', 0.5, 'FaceColor', 'red'); hold on
% [f,xi] = ksdensity(expt.trialsPresented.error(expt.trialsPresented.att==0));
% plot(xi,f, 'Color', 'red'); hold on
histogram(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==1)), hist_edges_deg(2:end), 'Normalization', 'probability', 'EdgeAlpha', 0, 'FaceAlpha', 0.5, 'FaceColor', 'blue'); hold on
xline(0, '--');
% xticks(xticks_hist);
legend({'Uncued targets', 'Cued targets', ''}, 'EdgeColor', [1 1 1]);
xlabel('Error (degrees)');
set(gcf, 'Color', [1 1 1]);
set(gca, 'TickLength', [0 0]);
set(gca, 'Box', 'off');
ax = get(gca, 'YAxis');
set(ax, 'Visible', 'off');
yticklabels({});



saveas(error, sprintf('%s/error_hist', dirname), ext);

% [f,xi] = ksdensity(expt.trialsPresented.error(expt.trialsPresented.att==1));
% plot(xi,f, 'Color', 'blue');

error_vs_uncertainty = figure;

scatter(abs(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==0))), expt.trialsPresented.arcAngleDeg(expt.trialsPresented.att==0), 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', 'red', 'MarkerFaceAlpha', 0.4); hold on
scatter(abs(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==1))), expt.trialsPresented.arcAngleDeg(expt.trialsPresented.att==1), 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', 'blue', 'MarkerFaceAlpha', 0.4); hold on

legend({'Uncued targets', 'Cued targets'}, 'EdgeColor', [1 1 1])

jn_plot_format();

xlabel('Absolute error (degrees)');
ylabel('Arc length (degrees)');
xticks([0:5:30])
yticks([0:10:50])
xlim([0, 30]);
ylim([0, 50]);

subjcorr = corr(abs(rad2deg(expt.trialsPresented.error))', expt.trialsPresented.arcAngleDeg');
title(sprintf('Subject %i, error-uncertainty correlation: %0.2g', i, subjcorr));

saveas(error, sprintf('%s/arc_scatter', dirname), ext);

end

%% summary plots 
summ_scatter = figure;
marker_types = {'o', 'p', '^'};
marker_colors = {'blue', 'red'};
for i = 1:size(expts,2)
    expt = expts(i).expt;
    
scatter(abs(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==0))), expt.trialsPresented.arcAngleDeg(expt.trialsPresented.att==0), 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', 'red', 'MarkerFaceAlpha', 0.3, 'Marker', marker_types{i}); hold on
scatter(abs(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==1))), expt.trialsPresented.arcAngleDeg(expt.trialsPresented.att==1), 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', 'blue', 'MarkerFaceAlpha', 0.3, 'Marker' ,marker_types{i}); hold on


end

jn_plot_format();

xlabel('Absolute error (degrees)');
ylabel('Arc length (degrees)');
xticks([0:5:30])
yticks([0:10:50])
xlim([0, 30]);
ylim([0, 50]);

xx = get(gca, 'Children');
xx = xx(1);
set(xx, 'EdgeColor', [0.5 0.5 0.5]);
set(xx, 'Box','off')

