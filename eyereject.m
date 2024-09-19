function exclude_trials = eyereject(eyedata, params)
exclude_trials = [];
event_times = eyedata.Events.Messages.time(cellfun(@(x) x=="EVENT_IMAGE",eyedata.Events.Messages.info))'-eyedata.timeline(1);

ntrials = numel(event_times);
blink = eyedata.Events.Eblink.start-eyedata.timeline(1);
sacc = eyedata.Events.Esacc.start-eyedata.timeline(1);

session_length = size(unique(eyedata.Samples.time),1);
raw_gaze = [eyedata.Samples.gx(:,1), eyedata.Samples.gy(:,1)];
l=size(raw_gaze,1);
x = reshape(raw_gaze(1:(l-mod(l,2)),:), 2, (session_length-1), 2);
gaze = squeeze(sum(x,1)/2);
gaze = [gaze; raw_gaze(end,:)];

blinkcount = 0;
sacccount = 0;

for i = 1:ntrials
    
    if any(blink>(event_times(i)+params.blinkwindow(1))&blink<(event_times(i)+params.blinkwindow(2)))
        blinkcount = blinkcount+1;
        exclude_trials = [exclude_trials, i];
        continue
    end

    if any(sacc>(event_times(i)+params.saccwindow(1))&sacc<(event_times(i)+params.saccwindow(2)))
        sacccount = sacccount+1;
        exclude_trials = [exclude_trials, i];
        continue
    end    

end
fprintf('%g trials excluded due to blinks', blinkcount);
fprintf('%g trials excluded due to saccades', sacccount);

idx = 1:ntrials;
idx(exclude_trials) = [];
maxdist = zeros(ntrials,1);
gwindsize = params.gazewindow(2)-params.gazewindow(1)+1;
for i = 1:numel(idx)
    trialid = idx(i);
    trialgaze = gaze((event_times(trialid)+params.gazewindow(1)):(event_times(trialid)+params.gazewindow(2)),:);
    maxdist(trialid) = max(sqrt(sum(trialgaze-repmat(params.center,[gwindsize,1]),2, 'omitnan').^2));
end

if ~isfield(params, 'distthresh')
    figure
    histogram(maxdist);
    title('maximum deviation of gaze from fixation in pixels for each trial');
    params.distthresh = input('set an upper threshold for distance from center');
end

gaze_exc = find(maxdist>params.distthresh);
fprintf('%g trials excluded due to poor fixation',numel(gaze_exc));
exclude_trials = [exclude_trials, gaze_exc'];

end