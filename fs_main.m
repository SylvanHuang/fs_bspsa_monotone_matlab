%
% Copyright, 2016.
% This program performs feature selection via
% Binary Simultanenous Perturbation Stochastic Approximation (BSPSA).
% For more information, please refer to the manuscript:
% "Feature Selection via Binary Simultaneous Perturbation Stochastic Approximation"
% By V. Aksakalli & M. Malekipirbazari, Pattern Recognition Letters, 2016.

clc; close all; clear all; warning('off', 'all')

global num_cv_reps 
global num_cv_folds 
global large_small_cutoff
global ds_name
global algo_wrapper 
global algo_fs 

large_small_cutoff = 100; % cut off for algo. parameters for large/ small datasets
num_cv_reps  = 1;
num_cv_folds = 5;

if isempty(gcp) 
   my_pool = parpool(num_cv_reps); 
end

% valid algo_fs options: bspsa, sfs, sbs
algo_fs = 'bspsa';   

% valid algo_wrapper options: knn, dt, svm
algo_wrapper = 'knn';  

% sample dataset is ionosphere with p = 34:
ds_name = 'ionosphere';
load ionosphere
Y = categorical(Y);
fs_manager(X,Y);

