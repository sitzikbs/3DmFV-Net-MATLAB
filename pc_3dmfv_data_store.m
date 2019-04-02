function [pcds] =pc_3dmfv_data_store(path, GMM, normalize, flatten)
% pc_3dmfv_data_store reutrns a point cloud data store for labled point
% clouds. It requires the directory names to be the labels.
%INPUT: path - string containing the path to directory which contains the labled subdirectories of point
%clouds
%OUTPUT: pcds - an image datastore object of 3dmfv representaiton

pcds = imageDatastore(path,...
    'ReadFcn',@pc_reader,...
    'FileExtensions','.txt',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

    function pc_3dmfv = pc_reader(filename)
        points = table2array(readtable(filename));
        points=Shrink2UnitSphere(points);
        pc_3dmfv = compute_3dmfv(points, GMM.w, GMM.mu, GMM.sigma, normalize, flatten);
    end
end


function [newPoints]=Shrink2UnitSphere(Points)
%Shrink2UnitSphere shrinks the given data x,y,z Points to fit insed a unit sphere
%and returns the new dataset
%INPUT : Points nx3
% move model to center of gepmetry
xyzmean=mean(Points, 1);
newPoints(:, 1)=Points(:, 1)-xyzmean(1);
newPoints(:, 2)=Points(:, 2)-xyzmean(2);
newPoints(:, 3)=Points(:, 3)-xyzmean(3);
%Shring the model
dist= sqrt(sum(newPoints.^2,2));
maxdist=max(dist);
newPoints=newPoints/(maxdist);
end