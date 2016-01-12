function fs_manager(X,Y)
% This function computes the performance metric with the full set of
% features and calls the specified feature selection method.


global ds_name
global algo_wrapper algo_fs
global num_feval large_small_cutoff

clc;
directory = './matlab_output/';
diary_file = strcat(directory, algo_fs, '_', algo_wrapper, '_', ds_name,'.txt');
diary(diary_file) % start diary

rng(1); % this doesn't work well for reproducibility because of the parfor loops

n = size(X,1);
p = size(X,2);

fprintf('FS Method: %s & Wrapper Classifier: %s\n', algo_fs, algo_wrapper);

fprintf('\nDataset: %s, n=%i, p=%i, c=%i \n##########################################\n',...
    ds_name, n, p, length(unique(Y)));

[err_mean, err_std, ~] = fs_loss(X, Y, ones(size(X,2),1));
fprintf('\nFull no. features = %i, error rate = %4.3f, error rate std = %4.3f\n',...
    p, err_mean, err_std);

tstart = tic;
num_feval = 0; % initialize to zero

switch algo_fs
   
    case 'bspsa'
        fs_spsa(X,Y);
        
    case 'sfs'
        if (p > large_small_cutoff) error('Error: Too many features.'); end
        fprintf('\n******* SFS: BEGIN ***********\n');
        [S, ~] = sequentialfs(@fs_seqsearch_func,X,Y,'cv','resubstitution','direction','forward');
        S = find(S>0);
        fprintf('\nSFS selected features:\n'); disp(S);
        [err_mean, err_std, ~] = fs_loss(X(:,S), Y, ones(size(X(:,S),2),1));
        fprintf('SFS no. features = %i, error rate = %4.3f, error rate std = %4.3f\n',...
            length(S), err_mean, err_std);
        fprintf('\n******* SFS: END ***********\n');
        
    case 'sbs'
        if (p > large_small_cutoff) error('Error: Too many features.'); end
        fprintf('\n******* SBS: BEGIN ***********\n');
        [S, ~] = sequentialfs(@fs_seqsearch_func,X,Y,'cv','resubstitution','direction','backward');
        S = find(S>0);
        fprintf('\nSBS selected features:\n'); disp(S);
        [err_mean, err_std, ~] = fs_loss(X(:,S), Y, ones(size(X(:,S),2),1));
        fprintf('SBS no. features = %i, error rate = %4.3f, error rate std = %4.3f\n',...
            length(S), err_mean, err_std);
        fprintf('\n******* SBS: END ***********\n');
        
    otherwise
        error('Error: Unknown FS method.');
end

fprintf('\nNumber of function evaluations = %i \nRun time = %1.2f secs\n\n', num_feval, toc(tstart));

diary off

end
