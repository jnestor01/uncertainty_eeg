
homefolder = '/Users/jeffreynestor/Desktop/Uncertainty/sim_data';
cd (homefolder);
parentfolder = 'sim_11_1_5';
cd (parentfolder);
iter = 1;
while 1
    if isfolder(sprintf('iter%s',num2str(iter)))
        filename = sprintf('iter%s/results.mat',num2str(iter));
        if isfile(filename)
            results(iter) = load(filename);
            iter = iter+1;
        else
            iter = iter-1;
            break
        end
    else
        iter = iter-1;
        break
    end
end

allerror = extractfield(results, 'errmat');

summary_error = mean(results(1:end).errmat);
summary_uncertainty =
summary_correlation = mean