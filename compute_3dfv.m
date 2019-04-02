function [pc_3dmfv] = compute_3dmfv(points, w, mu, sigma, normalize)
%compute_3dmfv computes the 3D modified Fisher Vector representation for a
%given point cloud
%INPUT:
%points : [ n_points x 3 x batch_size] batch point clouds (XYZ coordinates)
% gmm: a gaussian mixture model composed of : (listing dimensions for 3d but can be 2d as well) 
% w : [n_gaussians x 1] weights of a gaussian mixture model
% mu : [n_gaussians x 3 ]centers of a gaussian mixture model
% sigma : [1] std of a gaussian mixture model. (it is a scalar because of
% the 3dmfv definition, theoretically it can be a vector of 3 components
% per gaussian)
% OUTPUT: 
% pc_3dmfv [batch_size x 2d+1 x n_gaussians]  the 3dmfv representation for each
% point cloud in the batch

% w = gmm.ComponentProportion;
% mu = gmm.mu;
% sigma = gmm.Sigma;

if nargin < 5
    normalize = true;
end

n_batches = size(points, 3);
n_points = size(points, 1);
n_gaussians = size(mu, 2);
D = size(mu, 1);

batch_sigma = repmat(sigma, [1, 1, n_points, n_batches]);
batch_mu = repmat(mu, [1, 1, n_points, n_batches]);
batch_w = repmat(w, [1, 1, n_points, n_batches]);
batch_points = repmat(points, [1, 1, 1, n_gaussians] ); %n_points, D, n_batches, n_gaussians
batch_points = permute(batch_points, [2, 4, 1, 3]); % D, n_gaussians, npoints, n_batches

w_per_batch_per_d = repmat(w, [D, 1, n_batches]);  % 3D X n_gaussians X D (D for min and D for max and D for sum)
                                
p_per_point = (1.0 / (power(2.0 * pi, D / 2.0) .* power(sigma(1), D))) .* exp(-0.5 .* sum(( batch_points - batch_mu).^2 ./ batch_sigma, 1));
Q = p_per_point;
Q_per_d = repmat(p_per_point, [D, 1, 1, 1] ); 

sqrt_w = sqrt(batch_w);
d_pi_all = (Q - batch_w) ./ sqrt_w;
d_pi = reshape(sum(d_pi_all,3), [1, n_gaussians, n_batches]);

 d_mu_all = Q_per_d .* (batch_points - batch_mu) ./ batch_sigma;
 d_mu = (1 ./ sqrt(w_per_batch_per_d)) .* reshape(sum(d_mu_all, 3), [D, n_gaussians, n_batches]);

d_sig_all = Q_per_d .* (((batch_points - batch_mu).^2 ./ batch_sigma) - 1);
d_sigma = (1 ./ sqrt(2 * w_per_batch_per_d)) .* reshape(sum(d_sig_all, 3), [D, n_gaussians, n_batches]);                                                               

% number of points  normaliation
d_pi = d_pi / n_points;
d_mu = d_mu / n_points;
d_sigma =d_sigma / n_points;

if normalize
    % Power normaliation
    alpha = 0.5;
    d_pi = sign(d_pi) .* power(abs(d_pi), alpha);
    d_mu = sign(d_mu) .* power(abs(d_mu), alpha);
    d_sigma = sign(d_sigma) .* power(abs(d_sigma), alpha);
    
    % L2 normaliation
    d_pi = d_pi ./ repmat(sum(d_pi.^2, 2),[1, n_gaussians, 1]);
    d_mu = d_mu ./ repmat(sum(d_mu.^2, 2),[1, n_gaussians, 1]);
    d_sigma = d_sigma ./ repmat(sum(d_sigma.^2, 2),[1, n_gaussians, 1]);
end
d_pi(isnan(d_pi)) = 0;
d_mu(isnan(d_mu)) = 0;
d_sigma(isnan(d_sigma)) = 0;

pc_3dmfv = [d_pi; d_mu; d_sigma];

end

