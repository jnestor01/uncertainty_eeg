addpath(genpath('/Users/jeffreynestor/Desktop/cupcake/cupcake_data_EEG'));
filenames = {'cupcake_S0108_2cat', 'cupcake_S0004_2cat', 'cupcake_S0003_2cat'};
subjdirs = {'S0108', 'S0004', 'S0003'};

ext = 'png';

for i = 1:numel(filenames)
    expts(i) = load(filenames{i});
end

dims = [400 600];
legendboxalpha = 0.2;

scatters = figure('Position', [0, 0, dims(1), dims(2)*numel(filenames)]);
tiledlayout(numel(filenames), 1);

markeralpha = 0.3;
markercolors = {'blue', 'red'};

%% Individual plots
for i = 1:numel(filenames)
    nexttile;
    filename = filenames{i};
dirname = sprintf('%s/%s_behav_plots', subjdirs{i}, filename);
if ~exist(dirname, 'dir')
    mkdir(dirname);
end
expt = expts(i).expt;

scatter(abs(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==1))), expt.trialsPresented.arcAngleDeg(expt.trialsPresented.att==1), 'MarkerEdgeColor', markercolors{1}, 'MarkerEdgeAlpha', markeralpha, 'MarkerFaceColor', markercolors{1}, 'MarkerFaceAlpha', markeralpha); hold on

scatter(abs(rad2deg(expt.trialsPresented.error(expt.trialsPresented.att==0))), expt.trialsPresented.arcAngleDeg(expt.trialsPresented.att==0), 'MarkerEdgeColor', markercolors{2}, 'MarkerEdgeAlpha', markeralpha, 'MarkerFaceColor', markercolors{2}, 'MarkerFaceAlpha', markeralpha);hold on

if i==1
    leghandle = legend({'Cued targets', 'Uncued targets'}, 'EdgeColor', (1-legendboxalpha)*[1 1 1]);
end

jn_plot_format();

if i==numel(filenames)
xaxtext = xlabel('Absolute error (degrees)');
end
if i==round(numel(filenames)/2)
yaxtext = ylabel('Arc length (degrees)');
end
xticks([0:5:30])
yticks([0:10:50])
xlim([0, 30]);
ylim([0, 50]);

subjcorr = corr(abs(rad2deg(expt.trialsPresented.error))', expt.trialsPresented.arcAngleDeg');
title(sprintf('Subject %i', i));
subtitle(sprintf( 'Uncertainty-Error correlation: %0.2g',subjcorr));

end
loc = get(yaxtext, 'Position');
set(yaxtext, 'Position', loc+[-0.5, 0, 0])
loc = get(xaxtext, 'Position');
set(xaxtext, 'Position', loc+[0, -2, 0])

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


