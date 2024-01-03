clc;
clear all;
close all;

% 导入数据
fileID = fopen('./OriginData_735.txt', 'r');
delimiter = '\t';
data_cell = textscan(fileID, '%f%f%f%f%f', 'Delimiter', delimiter);
fclose(fileID);

data_origin = [data_cell{1:end}];
data = data_origin; % 加载数据，data包含73x5的数据矩阵，前四列为X，最后一列为Y
x = data(:, 1:end-1);  % 输入变量
y = data(:, end);      % 输出变量

% 数据集
dataset = data;

%% SVM参数
C = 20;
gamma = 1.7;

% 迭代次数
numIterations = 1000;

% 初始化结果矩阵
results = zeros(size(dataset, 1), numIterations + 1);

% 循环迭代
for i = 1:numIterations
    trainIndices = datasample(1:size(dataset, 1), 60, 'Replace', false);
    testIndices = setdiff(1:size(dataset, 1), trainIndices);
    
    % 归一化处理
    trainData = dataset(trainIndices, :);
    testData = dataset(testIndices, :);
    
    trainData_Normalized = (trainData - mean(trainData) )./ std(trainData);
    testData_Normalized = (testData - mean(trainData) )./ std(trainData);
    
    trainData_Z = trainData_Normalized;
    testData_Z = testData_Normalized;
    
    % 建立模型
    model = fitrsvm(trainData_Z(:, 1:4), trainData_Z(:, 5), 'KernelFunction', 'rbf', 'BoxConstraint', C, 'KernelScale', gamma);
    
    % 预测结果（反归一化）
    predictions_trainData_Z = predict(model, trainData_Z(:, 1:4));
    predictions_testData_Z = predict(model, testData_Z(:, 1:4));
    
%     for j = 1:length(predictions_trainData_Z)
%         results(trainIndices(j), i) = predictions_trainData_Z(j) * std(trainData(:, 5)) + mean(trainData(:, 5));
%     end
    
    for k = 1:length(predictions_testData_Z)
        results(testIndices(k), i) = predictions_testData_Z(k) * std(trainData(:, 5)) + mean(trainData(:, 5));
    end
end

%% 将非0值去除，组成全部由预测值的数组
% results中前1000列每一列都有13个数值，存放预测值，最后一列目前为空
[row, col] = size(results);
Results_nonZeroData = cell(row, 1);

for i = 1:73
    nonZeroValues = results(i, 1:col-1); % 取出每一行的除最后一列的数据
    nonZeroValues = nonZeroValues(nonZeroValues ~= 0); % 去除为零的值
    Results_nonZeroData{i} = nonZeroValues; % 存储去除零后的数据到单元数组中
    results(i, numIterations + 1) = mean(nonZeroValues); % 计算非零值的平均数
end

result_average = results(:, numIterations + 1);

% 写入excel，存在了sheet2中
tableData = cell2table(Results_nonZeroData);
filename = 'data0919.xlsx';
sheetname = 'Sheet3';
writetable(tableData, filename, 'Sheet', sheetname, 'WriteMode','append');
disp(result_average);
