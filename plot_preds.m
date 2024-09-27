function plt = plot_preds(posts, stimval, opts)

if nargin<3
    opts = struct();
end

if ~isfield(opts, 'binvals')
    opts.binvals = linspace(0,2*pi,size(posts,2)+1)';
    opts.binvals(end) = [];
end

if ~isfield(opts, 'alpha')
    %in theory, diameter of a point as a proportion of average side length
    %of the plot
    diam = 0.02;
    opts.alpha = min(1, (1/diam^2)/numel(stimval));
end
%posts: test trials x stimulus values
preds = zeros(size(posts,1),1);
    if strcmp(opts.outpmethod,'pop_vec')
        for i = 1:size(posts,1)
            pop_vec = posts(i,:)*exp(1i*opts.binvals); 
            preds(i) = angle(pop_vec);

        end
        preds(preds<0) = preds(preds<0)+(2*pi);
        preds = preds;
    end
% stimval = stimval/360*2*pi;
preds = preds/2/pi*360;
% mean(abs(circ_dist(stimval, preds)))
edges = 0:10:360;
figure
% plt = scatter(stimval, preds, 'filled', 'MarkerFaceAlpha', opts.alpha);
plt = histogram2(stimval/2/pi*360, preds, edges, edges, 'DisplayStyle','tile');

if ~isfield(opts, 'title')
    title('classifier predictions vs presented stimuli');
else
    title(opts.title, 'Interpreter', 'none')
end
xlabel('presented stimulus feature (deg)')
ylabel('classifier prediction (deg)')
cb = colorbar;
end