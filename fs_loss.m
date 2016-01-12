function [loss_mean, loss_std, loss_vec] = fs_loss(X,Y,wgt)
% This function computes the mean of repeated k-fold CV loss.
% wgt is a real-valued vector over [0,1]^p , which gets rounded here.

global num_cv_folds
global num_cv_reps

wgt_rounded = round(wgt)';

selected_features = nonzeros((1:size(X,2)) .* wgt_rounded);

X_fs = X(:,selected_features);  % X after feature selection

% mean of r-repeated f-fold CV:
loss_vec = zeros(num_cv_reps, 1);

% this rarely happens:
if (size(X_fs,2) == 0)
    loss_mean = mean(loss_vec);
    loss_std  = std(loss_vec);
    return
end

my_fit = fs_wrapper(X_fs,Y); % get the fit
parfor i=1:num_cv_reps
    cvmodel  = crossval(my_fit,'KFold',num_cv_folds);
    loss_vec(i,1) = kfoldLoss(cvmodel);
end

loss_vec = loss_vec';
loss_mean = mean(loss_vec);
loss_std  = std(loss_vec);

end

