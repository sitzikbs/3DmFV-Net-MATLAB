function [GMM] = get_2d_grid_gmm(n_gaussians, variance)
% compute_3dmfv computes the 3D modified Fisher Vector representation for a
% given point cloud
% INPUT:
% n_gaussians : [1 / 1x2] number of gaussians, can be a scalar or a vector of subdivisions 
% OUTPUT
% The gmm is composed of the following: 
% w : [1 x n_gaussians ] weights of a gaussian mixture model
% mu : [2 x n_gaussians]centers of a gaussian mixture model
% sigma : [2 x n_gaussians] covariances of a gaussian mixture model. 

if nargin == 1 
    variance = 0.04;
end

if length(n_gaussians) == 1
    n_gaussians = [n_gaussians, n_gaussians];
end
n_gaussians_total = n_gaussians(1) * n_gaussians(2);
w = ones(1, n_gaussians_total);
step = [ 1 / n_gaussians(1), 1 / n_gaussians(2)];
[mu_x, mu_y] = meshgrid(linspace(-1 + step(1),1-step(1), n_gaussians(1)),...
    linspace(-1 + step(2),1-step(2), n_gaussians(2)));
mu = [mu_x(:), mu_y(:)]';
sigma = sqrt(variance) * ones(size(mu));

% sigma = [variance, variance] ;
% gmm = gmdistribution(mu, sigma, w);
GMM.w = w;
GMM.mu = mu;
GMM.sigma = sigma;
end

