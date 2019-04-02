
function [pcds] = point_cloud_data_store(path)
% point_cloud_data_store reutrns a point cloud data store for labled point
% clouds. It requires the directory names to be the labels.
%INPUT: path - string containing the path to directory which contains the labled subdirectories of point
%clouds
%OUTPUT: pcds - an image datastore object of point clouds. 

pcds = imageDatastore(path,...
                                        'ReadFcn',@pc_reader,...
                                         'FileExtensions','.txt',...
                                         'IncludeSubfolders',true,...
                                         'LabelSource','foldernames');                                 
end


function data = pc_reader(filename)
data = table2array(readtable(filename));
end