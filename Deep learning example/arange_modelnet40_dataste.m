Drdfunction arange_modelnet40_dataset()
clear all
close all
clc

dataset_path = 'C:\Users\Itzik\Documents\Datasets\ModelNet40\';
relative_path = 'data\modelnet40_ply_hdf5_2048\';
train_file_list = [dataset_path, relative_path, 'train_files.txt'];
test_file_list = [dataset_path, relative_path, 'test_files.txt'];
shape_names_file = [dataset_path, relative_path, 'shape_names.txt'];

output_dir = [dataset_path, 'matlab_dataset\'];
out_test_path  = [output_dir, 'test\'];
out_train_path = [output_dir, 'train\'];
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    mkdir(out_test_path);
    mkdir(out_train_path);
end

test_file_names = get_file_list(test_file_list, dataset_path);
train_files_names=get_file_list(train_file_list, dataset_path);
shape_names=get_file_list(shape_names_file, '');
for i =1:length(shape_names)
    shape = shape_names{i};
    if ~exist([out_test_path, shape], 'dir')
        mkdir([out_test_path, shape])
    end
    if ~exist([out_train_path, shape],'dir')
        mkdir([out_train_path, shape])
    end
end
extract_data(test_file_names, out_test_path, shape_names);
extract_data(train_files_names, out_train_path, shape_names);
end

function files=get_file_list(file_list, dataset_path)
fid=fopen(file_list);
files = cell(0,1);
line = fgetl(fid);
while ischar(line)
    files{end+1,1} = [dataset_path, line];
    line = fgetl(fid);
end
fclose(fid);
end

function extract_data(file_names, output_path, shape_names)
shape_counters =ones(1,length(shape_names));

for i=1:length(file_names)
    file_name = file_names{i};
    data = hdf5read(file_name,'data');
    labels = hdf5read(file_name,'label');
    for j=1:size(data,3)
        points = data(: ,:, j);
        label_txt = shape_names{labels(j)+1};
        output_file_name = [output_path, label_txt,'\', num2str(shape_counters(labels(j)+1)),'.txt'];
        shape_counters(labels(j)+1)= shape_counters(labels(j)+1)+1;
        dlmwrite(output_file_name, points')
    end
end
end
