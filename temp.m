clear all
close all
clc

n_gaussians = 2;
variance = 1;
normalize = true;
path = 'C:\Users\Itzik\Documents\Datasets\Elbit\Classification\Real data - labled by elbit\test_set';
[w, mu, sigma] = get_3d_grid_gmm(n_gaussians, variance);

[pcds] =pc_3dmfv_data_store(path, w, mu, sigma, normalize);

fv_test = readimage(pcds,1);
disp('DONE');

%waiting for MATLAB to support 3d convolutions