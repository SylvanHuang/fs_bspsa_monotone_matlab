function [loss_mean, loss_std, loss_vec] = ...
    fs_bspsa_loss_func(X, Y, w, num_cv_reps, num_cv_folds)
% returns the mean of r-repeated f-fold CV error rate

w_rounded = round(w)';

selected_features = nonzeros((1:size(X,2)) .* w_rounded);

X_fs = X(:,selected_features);

my_fit = fs_bspsa_wrapper(X_fs,Y); 

loss_vec_reps = zeros(num_cv_reps, 1);
parfor i = 1:num_cv_reps
   cvmodel = crossval(my_fit,'KFold',num_cv_folds);
   loss_vec_reps(i,1) = kfoldLoss(cvmodel); 
end

loss_vec  = loss_vec_reps';
loss_mean = mean(loss_vec_reps);
loss_std  = std(loss_vec_reps);

end

