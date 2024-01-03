clc;
clear all;
close all;

%% 
% 1005下午
% 神经网络参数：1隐藏层神经元数量-7；2学习率-0.01
% 用于扩容材料数据的确定

%% 数据输入
% Data : 1-总应变幅度；2-塑性应变幅度；3-最大应力；4-温度；5-疲劳寿命
file1 = './OriginData_Train1285.txt';
train1 = dlmread(file1, '\t');

file2 = './OriginData_Test335.txt';
test1 = dlmread(file2, '\t');

%% 数据集已提前手动划分
trainData = train1;
testData = test1;

trainData_Z = (trainData - mean(trainData) )./ std(trainData);

% 测试集的归一化同样采用训练集的参数【防止数据泄露】
testData_input_Z = (testData - mean(trainData) )./ std(trainData);

%% net
% 创建神经网络
% net = newff(trainData_Z(:, 1:4), trainData_Z(:, 5), [6, 4]);
net = newff(trainData_Z(:, 1:4)', trainData_Z(:, 5)', 7);
    
% 设置训练参数
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;

% 学习率
net.trainParam.lr = 0.01;

net.trainParam.max_fail = 6;
net.trainParam.showWindow = true;

net.divideParam.trainRatio = 0.85;   %训练集占比
net.divideParam.valRatio = 0.15;      %验证集占比
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
   
%% 绘图
% 训练集合
% figure
% plot(1:165, trainData(:, 5), 'r-*', 1:165, predictions_trainData_Z, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% title('训练样本')
% grid

figure
plot(1:size(predictions_testData_Z, 1), testData(:, 5), 'r-*', 1:size(predictions_testData_Z, 1), predictions_testData_Z, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
title('测试样本')
grid

% 观测指标
x1 = predictions_testData_Z;
x2 = testData(:, 5);
corr_matrix = corrcoef(x1, x2);
correlation = corr_matrix(1, 2); % 获取相关系数
R_squared = correlation^2;
disp(correlation);
    

%% 保存网络
% save('BPNN5.mat', 'net');
% load('BPNN1.mat');
% net_testData_Z = sim(net, testData_input_Z(:, 1:4)');
% net_testData_Z = net_testData_Z' * std(trainData(:, 5)) + mean(trainData(:, 5));
