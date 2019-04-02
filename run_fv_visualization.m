clear all
close all
clc

n_gaussians = 1;
variance = 1;

[w, mu, sigma] = get_2d_grid_gmm(n_gaussians, variance);

points = [0, 0.3;0, -0.3];
% points(:,:,2) =[1, 1];
% points = rand(1024, 2, 128);
pc_3dmfv = compute_3dmfv(points, w, mu, sigma, false);

visualize_2d_3dmfv(points, w, mu, sigma, pc_3dmfv)