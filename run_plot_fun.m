load('results_timecat.mat');
partition.type = 'cond';
partition.cond = 'regressionmethod';
% partition.type = 'lines';
% partition.cond = {'train_subset'};
% partition.lines = {{'0.5'}};
% partition.cond = {'spatialLocation'};
% partition.lines = {{'[90,270]'}, {'[180,0]'}};
% partition.type = 'stim';
% partition.cond = 'gratingContrast';
opts.plot_conds_array = [1];
% opts.type.errflag = 0;
% opts.type.uncflag = 0;
% opts.type.corrflag = 0;
% opts.type.gammaflag = 0;
% opts.type.predflag = 0;
% opts.type.scatterflag= 0;
% opts.type.behavflag = 0;
% opts.type.errflag = 0;
% opts.type.uncflag = 0;
% opts.type.corrflag = 0;
% opts.type.gammaflag = 0;
opts.type.predflag = 1;
opts.type.scatterflag= 1;
opts.pred.nbins = 12;


opts.type.errflag = 1;
opts.type.uncflag = 1;
opts.type.corrflag = 1;
opts.type.gammaflag = 0;
opts.type.predflag = 1;
opts.type.scatterflag= 1;
opts.type.behavflag = 1;
opts.type.errflag = 1;
opts.type.uncflag = 1;
opts.type.corrflag = 1;
opts.type.gammaflag = 1;
% opts.iterselect.field = 'ntraintrials';
% opts.iterselect.value = 600;
opts.ci = 0;
opts.significance = 0;
opts.statalpha = 0.05;
opts.norm_unc = 0;

opts.plt.x0=10;
opts.plt.y0 = 10;
opts.plt.width = 1000;
opts.plt.height = 600;
opts.plt.cialpha = 0.1;
opts.plt.linewidth = 1;
% opts.plt.titlesuffix = ' for low contrast training and testing';
opts.plt.titlesuffix = '';
opts.plt.nticks = 20;
opts.null = 0;
% opts.pred.type = 'cardinalsvsoblique';
opts.pred.type = 'reg';
% opts.pred.type = 'prior';
opts.pred.rank = 0;

opts.pred.outpmethod = 'pop_vec';
% opts.pred.timepoint = 95;
% opts.pred = rmfield(opts.pred, 'timepoint');

jn_plot_fun(results,partition,opts);

opts.type.errflag = 0;
opts.type.uncflag = 0;
opts.type.corrflag = 0;

for timepoint = [1:5:60]
    opts.pred.timepoint = timepoint;
    jn_plot_fun(results,partition,opts);
end
