function my_fit = fs_wrapper(X, Y)
% This function specifies the wrapper classifier to be used.
% It also increments the global variable 'num_feval' 
% for each objective function evaluation.

global algo_wrapper
global num_feval
num_feval = num_feval +1;

switch algo_wrapper
   case 'knn'
      my_fit = fitcknn(X,Y,'NumNeighbors',1,'Standardize',1);
      
   case 'dt'
      my_fit = fitctree(X,Y);
      
   case 'svm'
      t = templateSVM('KernelFunction','linear','Standardize',1);
      my_fit = fitcecoc(X,Y,'Learners',t);
      
   otherwise
      error('Error: Unknown FS wrapper.');
end

end

