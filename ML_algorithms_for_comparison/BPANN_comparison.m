clc;
clear all;
close all;

% 导入数据
fileID = fopen('./OriginData_735.txt', 'r');
delimiter = '\t';
data_cell = textscan(fileID, '%f%f%f%f%f', 'Delimiter', delimiter);
fclose(fileID);

data_origin = [data_cell{1:end}];
dataset = data_origin; % 加载数据，data包含73x5的数据矩阵，前四列为X，最后一列为Y
x = dataset(:, 1:end-1);  % 输入变量
y = dataset(:, end);      % 输出变量

% 迭代次数
numIterations = 1000;

% 初始化结果矩阵
results = zeros(size(dataset, 1), numIterations + 1);

%% 训练集划分
% 随即划分训练集（60个）、测试集（13个）
for i = 1:numIterations
    trainIndices = datasample(1:size(dataset, 1), 60, 'Replace', false);
    testIndices = setdiff(1:size(dataset, 1), trainIndices);
    
    % 归一化处理
    trainData = dataset(trainIndices, :);
    testData = dataset(testIndices, :);
    
    trainData_Z = (trainData - mean(trainData) )./ std(trainData);
    
    % 测试集的归一化同样采用训练集的参数【防止数据泄露】
    testData_input_Z = (testData - mean(trainData) )./ std(trainData);
    
    %% 创建神经网络
    % 输入数据为：4*60 + 1*60 + 隐含层神经元为X （连接权值NumWeightElements有4*X+X*1+X+1=6X+1）
    % net = newff(trainData_Z(:, 1:4), trainData_Z(:, 5), [6, 4]);
    net = newff(trainData_Z(:, 1:4)', trainData_Z(:, 5)', 7);
    
    % 设置训练参数
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.01;
    net.trainParam.max_fail = 15;
    net.trainParam.showWindow = false;

    net.divideParam.trainRatio = 0.8;   %训练集占比
    net.divideParam.valRatio = 0.2;      %验证集占比
    net.divideParam.testRatio = 0;     %测试集占比

    % 训练网络
    [net, tr] = train(net, trainData_Z(:, 1:4)', trainData_Z(:, 5)');
    
    %% 结果
    % 预测
    predictions_trainData_Z = sim(net, trainData_Z(:, 1:4)');
    predictions_testData_Z = sim(net, testData_input_Z(:, 1:4)');
     
    %反归一化
    predictions_trainData_Z = predictions_trainData_Z' * std(trainData(:, 5)) + mean(trainData(:, 5));
    predictions_testData_Z = predictions_testData_Z' * std(trainData(:, 5)) + mean(trainData(:, 5));
       
    %填入数组
%     for j = 1:length(predictions_trainData_Z)
%         results(trainIndices(j), i) = predictions_trainData_Z(j);
%     end
    
    for k = 1:length(predictions_testData_Z)
        results(testIndices(k), i) = predictions_testData_Z(k);
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

% % 写入excel，存在了sheet1中
% tableData = cell2table(Results_nonZeroData);
% filename = 'data0920.xlsx';
% sheetname = 'Sheet1';
% writetable(tableData, filename, 'Sheet', sheetname);
% disp(result_average);

% 写入excel，存在了sheet2中，用的是append
% tableData = cell2table(Results_nonZeroData);
% filename = 'data0920.xlsx';
% sheetname = 'Sheet3';
% writetable(tableData, filename, 'Sheet', sheetname, 'WriteMode','append');
% disp(result_average);

% % 绘图
% figure
% plot(1:13, testData(:, 5), 'r-*', 1:13, predictions_testData_Z, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% grid
