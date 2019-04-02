function [h] = visualize_2d_3dmfv(points, w, mu, sigma, pc_3dmfv)
% visualize_2d_3dmfv 
figh = figure('color','w');
axh = axes('xlim', [-1.2, 1.2], 'ylim',[-1.2, 1.2]);
daspect([1,1,1]);
circles_h = viscircles( mu', sigma(1,:), 'linestyle','--','color','k' );
hold all
points_h = scatter(points(:, 1), points(:, 2),'marker','o');

end

