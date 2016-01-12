function [selected_features, overall_best_fval_mean, overall_best_fval_stdev, overall_best_reps] = ...
   fs_spsa(X,Y)
% This function runs binary SPSA for feature selection.

global large_small_cutoff num_feval num_cv_reps

print_output = 1;
p = size(X,2);

c = 0.05;
alpha = 0.6;
num_runs = 1;
   
if (p < large_small_cutoff) %Best: A=100, a=0.75, c=0.1
   A = 100;
   a = 0.75;
   num_iters = 1000;
else
   A = 300;
   a = 1.5;
   num_iters = 3000;
end

iter_stall_limit = round(num_iters * 0.25);
stall_tolerance = 1/10^5;

fprintf('\n******* SPSA: BEGIN ***********\n');
num_feval = 0; % initialize to zero

% for each spsa run, spsa_run_results stores the following:
% wgt_opt, min_fval_mean, min_fval_stdev, min_index, individual CV rep. means
spsa_run_results = zeros(num_runs, p+3+num_cv_reps);

% for each spsa run, spsa_loss_mat stores the run's iteration loss values
% it is stored as a mat file for further analysis
spsa_loss_mat = zeros(num_iters, num_runs);

wgt_min = 0 * ones(p,1);
wgt_max = 1 * ones(p,1);

for i = 1:num_runs
   
   % wgt_vec stores the spsa weights at each iteration
   wgt_vec = zeros(num_iters+1,p);
   
   % fval_vec stores the mean & stdev. and
   % cv repetition values of r-repeated f-fold CV loss at each iteration
   fval_vec = zeros(num_iters+1, 2+num_cv_reps);
   
   % initial weights
   curr_wgt = 0.5 * ones(p,1);
   
   wgt_vec(1,:) = curr_wgt;
   
   if (print_output == 1)
      fprintf('\nSPSA Run No %i:\n*********************\n',i);
   end
   
   [fval_vec(1,1), fval_vec(1,2), fval_vec(1,3:(2+num_cv_reps))] = fs_loss(X, Y, curr_wgt);
   
   if (print_output == 1)
      fprintf('k = 0, fval mean = %4.3f, fval stdev = %4.3f\n', fval_vec(1,1), fval_vec(1,2));
   end
   
   best_fval = 99999;
   stall_counter = 0;
   
   for k = 1:num_iters
      
      ak = a/(k+A)^alpha;
      
      delta = 2 * round(rand(p,1)) - 1; % random +1 or -1
      
      wgt_plus = curr_wgt + c*delta;
      wgt_plus = max(wgt_plus, wgt_min);
      wgt_plus = min(wgt_plus, wgt_max);
      
      wgt_minus  = curr_wgt - c*delta;
      wgt_minus = max(wgt_minus, wgt_min);
      wgt_minus = min(wgt_minus, wgt_max);
      
      yplus  = fs_loss(X, Y, wgt_plus);
      yminus = fs_loss(X, Y, wgt_minus);
      
      ghat = (yplus - yminus) ./ (2*c*delta);
      
      curr_wgt = curr_wgt - ak*ghat;
      curr_wgt = min(curr_wgt, wgt_max);
      curr_wgt = max(curr_wgt, wgt_min);
      
      wgt_vec(k+1,:) = curr_wgt;
      [fval_vec(k+1,1), fval_vec(k+1,2), fval_vec(k+1,3:(2+num_cv_reps))] = fs_loss(X, Y, curr_wgt);
      
      spsa_loss_mat(k,i) = fval_vec(k+1,1);
      
      if ((print_output == 1) && (mod(k,10) == 0))
         selected_features = nonzeros((1:p) .* round(curr_wgt'));
         fprintf('k = %i, fval mean = %4.3f, fval stdev = %4.3f (%i features)\n',...
            k, fval_vec(k+1,1), fval_vec(k+1,2), length(selected_features));
      end
      
      if(fval_vec(k+1,1) < best_fval - stall_tolerance)
         best_fval = fval_vec(k+1,1);
         stall_counter = 0;
      else
         stall_counter = stall_counter+1;
      end
      
      if (stall_counter > iter_stall_limit)
         break;
      end
      
   end
   
   [min_fval_mean, min_index] = min(fval_vec(1:k,1)); % find the best weight
   
   min_fval_stdev = fval_vec(min_index,2);
   
   wgt_opt_rounded = round(wgt_vec(min_index,:));
   
   spsa_run_results(i,1:p) = wgt_opt_rounded;
   spsa_run_results(i,p+1) = min_fval_mean;
   spsa_run_results(i,p+2) = min_fval_stdev;
   spsa_run_results(i,p+3) = min_index;
   spsa_run_results(i,p+4:(p+3+num_cv_reps)) = fval_vec(min_index,3:(2+num_cv_reps));
   
   if (print_output == 1)
      fprintf('\nSPSA run min error: mean: %4.3f, stdev: %4.3f, index: %i\n',...
         min_fval_mean, min_fval_stdev, min_index-1);
   end
   
end

[overall_best_fval_mean, overall_best_index] = min(spsa_run_results(:,p+1));
overall_best_fval_stdev = spsa_run_results(overall_best_index,p+2);
overall_best_wgt_vec_rounded = spsa_run_results(overall_best_index,1:p);
selected_features = nonzeros((1:p) .* overall_best_wgt_vec_rounded);
overall_best_reps = spsa_run_results(overall_best_index,p+4:(p+3+num_cv_reps));

if (p < large_small_cutoff)
   fprintf('\nSPSA selected %i features:\n', length(selected_features'));
   disp(selected_features');
end

fprintf('\nOverall best SPSA no. features = %i, error rate = %4.3f, error rate std = %4.3f\n',...
   length(selected_features), overall_best_fval_mean, overall_best_fval_stdev);

plot(smooth(fval_vec(:,1))) % plot the smoothened latest spsa run

fprintf('\n******* SPSA: END ***********\n');

end
