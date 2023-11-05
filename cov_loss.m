function loss = cov_loss(est_cov, samp_cov)
        try
            loss = (logdet(est_cov, 'chol') + sum(sum(invChol_mex(est_cov).*samp_cov)))/size(samp_cov,2);
        catch ME
            if any(strcmpi(ME.identifier, {'MATLAB:posdef', 'MATLAB:invChol_mex:dpotrf:notposdef'}))
                loss = (logdet(est_cov) + trace(est_cov\samp_cov))/size(samp_cov,2);
            else
                rethrow(ME);
            end
        end
        if imag(loss)~=0, loss = inf; end
end