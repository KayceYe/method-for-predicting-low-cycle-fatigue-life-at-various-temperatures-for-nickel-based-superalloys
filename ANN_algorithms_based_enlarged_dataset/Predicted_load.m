clc
clear all
close all

%% Load dataset
% file_test = './Dataset_FGH95.txt';
% file_test = './Dataset_TC4.txt';
file_test = './Dataset_Waspaloy.txt';
testData = dlmread(file_test, '\t');

% Z-score normalization
mean_trainData = [0.705207812500000, 0.243781552820313, 844.398437500000, 744.321875000000, 12225.7968750000];
std_trainData = [0.317280917683188, 0.264784098644345, 243.273390159598, 174.060452408121, 17362.2582012876];
testData_input_Z = (testData - mean_trainData)./ std_trainData;

%% Calculate
% Load neural network, named 'net'
% load('Predicted_GABP.mat'); % Predicted by GA-BP
load('Predicted_BPANN.mat'); % Predicted by BP-ANN

% Function:Sim
predictions_testData_Z = sim(net, testData_input_Z(:, 1:4)');

% Prediction
b = 1.736225820128762e+04;
c = 1.222579687500000e+04;
predictions_testData_Z = predictions_testData_Z' * std_trainData(5) + mean_trainData(5);

%% Plot
figure
plot(1:size(predictions_testData_Z, 1), testData(:, 5), 'r-*', 1:size(predictions_testData_Z, 1), predictions_testData_Z, 'b-o', 'LineWidth', 1)
% legend('', '')
xlabel('Sr.no')
ylabel('Experimental/Predicted fatigue life')
title('Waspaloy')
grid

% Performance
x1 = predictions_testData_Z;
x2 = testData(:, 5);
corr_matrix = corrcoef(x1, x2);
correlation = corr_matrix(1, 2); 
R_squared = correlation^2;
disp(correlation);
