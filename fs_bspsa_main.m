%
% Copyright, 2015
% This program performs feature selection via
% Binary Simultanenous Perturbation Stochastic Approximation (BSPSA).
% The benchmark dataset used is the Ionosphere dataset.
% For more information, please refer to the manuscript:
% "Feature Selection via Binary Simultaneous Perturbation Stochastic Approximation"
% By V. Aksakalli & M. Malekipirbazari
% arXiv.org ID:
% If you are using this code for research, please cite it as 
% Submitted for Publication, Pattern Recognition Letters.

clc; close all; clear;

load ionosphere % 34 features
Y = categorical(Y);

p = size(X,2);

num_cv_reps  = 10;
num_cv_folds = 5;

% Recommended values for datasets with less than 100 features:
num_bspsa_runs  = 5;
num_bspsa_iters = 1000;
A = 100;
a = 1.0;
alpha = 0.6;
%%%%%%%%%%%%%%%%%%%

% Recommended values for datasets with more than 100 features:
% % num_bspsa_runs  = 1;
% % num_bspsa_iters = 3000;
% % A = 300;
% % a = 2.0;
% % alpha = 0.6;
%%%%%%%%%%%%%%%%%%%

w_0_low  = 0.4;
w_0_high = 0.6;
w_lo = 0 * ones(p,1);
w_hi = 1 * ones(p,1);

X = zscore(X); % standardize X with zero mean and 1 std

err_vec_reps = zeros(num_cv_reps,1);

% compute average cross-validation error rate for the full feature set:
my_fit = fs_bspsa_wrapper(X,Y);
parfor i = 1:num_cv_reps
    err_vec_reps(i,1) = kfoldLoss(crossval(my_fit,'KFold',num_cv_folds));
end
fprintf('Full feature set error: mean = %f, std = %f (with %i features)\n\n',...
    mean(err_vec_reps), std(err_vec_reps), p);

% bspsa_run_results stores the following for each BSPSA run:
% w_opt, min_loss_mean, min_loss_stdev, min_loss_index
bspsa_run_results = zeros(num_bspsa_runs, p+3);

% perform the BSPSA runs:
for i = 1:num_bspsa_runs
    
    t = cputime; % to measure execution time for each run
    
    % sol_vec stores the solution vector at each iteration
    sol_vec = zeros(num_bspsa_iters+1,p);
    
    % iter_vec stores the mean & std and
    % CV repetition values of r-repeated f-fold CV loss at each iteration
    iter_vec = zeros(num_bspsa_iters+1,2+num_cv_reps);
    
    curr_w = w_0_low + (w_0_high - w_0_low)*rand(p,1); % initial weights
    sol_vec(1,:) = curr_w;
    
    fprintf('BSPSA Run No %i:\n************************\n',i);
    
    [iter_vec(1,1), iter_vec(1,2), iter_vec(1,3:(2+num_cv_reps))] = ...
        fs_bspsa_loss_func(X, Y, curr_w, num_cv_reps, num_cv_folds);
        
    fprintf('k = 0, error: mean = %f, std = %f\n', iter_vec(1,1), iter_vec(1,2));
       
    for k = 1:num_bspsa_iters
        
        ak = a/(k+A)^alpha;
        
        delta = 2 * round(rand(p,1)) - 1;
        
        w_plus = curr_w + delta/2;
        w_plus = max(w_plus, w_lo);
        w_plus = min(w_plus, w_hi);
        
        w_minus  = curr_w - delta/2;
        w_minus = max(w_minus, w_lo);
        w_minus = min(w_minus, w_hi);
        
        yplus  = fs_bspsa_loss_func(X, Y, w_plus, num_cv_reps, num_cv_folds);
        yminus = fs_bspsa_loss_func(X, Y, w_minus, num_cv_reps, num_cv_folds);
        
        ghat = (yplus - yminus) ./ delta;
        
        curr_w = curr_w - ak*ghat;
        
        curr_w = min(curr_w, w_hi);
        curr_w = max(curr_w, w_lo);
        
        sol_vec(k+1,:) = curr_w;
        
        [iter_vec(k+1,1), iter_vec(k+1,2), iter_vec(k+1,3:(2+num_cv_reps))] = ...
            fs_bspsa_loss_func(X, Y, curr_w, num_cv_reps, num_cv_folds);
        
        fprintf('k = %i, error: mean = %f, std = %f\n',...
            k, iter_vec(k+1,1), iter_vec(k+1,2));
        
    end
    
    [min_loss_mean, min_index] = min(iter_vec(:,1)); % find the best weight
    
    min_loss_stdev = iter_vec(min_index,2);
    
    w_opt_rounded = round(sol_vec(min_index,:));
    
    bspsa_run_results(i,1:p) = w_opt_rounded;
    bspsa_run_results(i,p+1) = min_loss_mean;
    bspsa_run_results(i,p+2) = min_loss_stdev;
    bspsa_run_results(i,p+3) = min_index;
      
    t = cputime-t;
    
    fprintf('\nRun''s min error = mean: %f, std = %f, index: %i (exec. time: %f secs) \n\n',...
        min_loss_mean, min_loss_stdev, min_index-1, t);
        
end

[overall_best_loss_mean, overall_best_index] = min(bspsa_run_results(:,p+1));

overall_best_loss_stdev = bspsa_run_results(overall_best_index,p+2);

overall_best_w_vec_rounded = bspsa_run_results(overall_best_index,1:p);

selected_features = nonzeros((1:p) .* overall_best_w_vec_rounded);

fprintf('Summary of BSPSA Runs:\n************************\n');

fprintf('Overall min error: mean = %f, std = %f (with %i features)\n\n', ...
    overall_best_loss_mean, overall_best_loss_stdev, length(selected_features));

fprintf('Features selected by BSPSA:\n');
disp(selected_features');
