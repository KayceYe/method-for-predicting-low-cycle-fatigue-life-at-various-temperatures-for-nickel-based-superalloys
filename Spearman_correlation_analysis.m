clc;
clear all;
close all;

% 1-total strain；2-elsatic strain；3-plastic strain；4-stress amplitude，5-Young's modulus；6-Temperature；7-LCF life
fileID = fopen('./OriginData_73.txt', 'r');
delimiter = '\t';
data_cell = textscan(fileID, '%f%f%f%f%f%f%f', 'Delimiter', delimiter);
fclose(fileID);

% Import ranked value
data_origin = [data_cell{1:end}];
data = data_origin;
X = data(:, 1:end-1);  %
Y = data(:, end);      


correlation_matrix = corrcoef([X Y]); 
pearson_coefficients = correlation_matrix(1:6, 7); 

disp(pearson_coefficients);

%% plot
figure;
XDisplayLabels = {'TS', 'ES', 'PS', 'ST', 'E', 'T', 'LCF'};
YDisplayLabels = {'TS', 'ES', 'PS', 'ST', 'E', 'T', 'LCF'};
h = heatmap(correlation_matrix, 'Colormap', bone(256), 'ColorbarVisible', 'on', 'XDisplayLabels', XDisplayLabels, 'YDisplayLabels', YDisplayLabels);
set(gca, 'FontSize', 18, 'FontName', 'Times New Roman');
set(h, 'GridVisible', 'off');


