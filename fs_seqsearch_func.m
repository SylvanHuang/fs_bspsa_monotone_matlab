function [test_err] = fs_seqsearch_func(X_train, Y_train, ~, ~)
% This function is needed by SFS and SBS

test_err = fs_loss(X_train, Y_train, ones(size(X_train,2),1));

end

