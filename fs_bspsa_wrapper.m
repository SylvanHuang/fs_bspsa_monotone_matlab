function [my_fit] = fs_bspsa_wrapper(X, Y)
% the wrapper classifier is specified here.

my_fit = fitcknn(X,Y,'NumNeighbors',1);

%my_fit = fitcecoc(X,Y,'Learners',templateSVM('KernelFunction','linear'));

%my_fit = fitcnb(X,Y,'DistributionNames','kernel');

%my_fit = fitctree(X,Y);

end

