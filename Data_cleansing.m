clc;
clear all;
close all;

% fopen、fclose
fileID = fopen('./OriginData_87.txt', 'r');
delimiter = '\t';
data_cell = textscan(fileID, '%f%f%f%f%f%f%f', 'Delimiter', delimiter);
fclose(fileID);

% Dataset separation
data_origin = [data_cell{1:end}];
data = data_origin; 
X = data(:, 1:end-1);  
Y = data(:, end);  

meds = median(data);
MADs = mad(data);
bbb = abs(data - meds) ./ MADs;
aaa = sum(abs(data - meds) ./ MADs);
distances = sum(abs(data - meds) ./ MADs, 2);

threshold = median(distances) + 3 * mad(distances);
outliers = find(distances > threshold);
disp(outliers);

num_samples = size(data, 1);

is_outlier = false(num_samples, 1);
is_feature_outlier = distances > threshold;
is_outlier = is_outlier | is_feature_outlier;

data_cleaned = data(~is_outlier, :);