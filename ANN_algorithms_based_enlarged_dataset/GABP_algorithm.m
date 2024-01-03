clc;
clear all;
close all;

tic

%%
% 1008下午
% 隐藏层神经元数量-7；2学习率-0.01
% 1-总应变幅度；2-塑性应变幅度；3-最大应力；4-温度；5-疲劳寿命
file1 = './OriginData_Train1605.txt';
train1 = dlmread(file1, '\t');

file2 = './OriginData_Test395.txt';
test1 = dlmread(file2, '\t');

%% 训练集划分
% 输入层、隐藏层、输入层节点个数
inputnum = 4;
hiddennum = 7;
outputnum = 1;


%% 数据集已提前手动划分
trainData = train1;
testData = test1;

trainData_Z = (trainData - mean(trainData) )./ std(trainData);

% 测试集的归一化同样采用训练集的参数【防止数据泄露】
testData_input_Z = (testData - mean(trainData) )./ std(trainData);

% 接入GA-BP
inputn =  trainData_Z(:, 1:4)';
outputn = trainData_Z(:, 5)';

%% net
% 创建神经网络
% 输入数据为：4*60 + 1*60 + 隐含层神经元为X （连接权值NumWeightElements有4*X+X*1+X+1=6X+1）
% net = newff(trainData_Z(:, 1:4), trainData_Z(:, 5), [6, 4]);
net = newff(inputn, outputn, hiddennum);

%% 遗传算法参数初始化
maxgen=80;                         %进化代数，即迭代次数
sizepop=30;                        %种群规模
pcross=0.4;                       %交叉概率选择，0和1之间
pmutation=0.1;                    %变异概率选择，0和1之间

% 节点总数
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

lenchrom=ones(1,numsum);       
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %数据范围

individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %将种群信息定义为一个结构体
avgfitness=[];                      %每一代种群的平均适应度
bestfitness=[];                     %每一代种群的最佳适应度
bestchrom=[];                       %适应度最好的染色体

for i=1:sizepop                                  %随机产生一个种群
    individuals.chrom(i,:)=Code(lenchrom,bound);    %编码
    x=individuals.chrom(i,:);                     %计算适应度
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %染色体的适应度
end

[bestfitness, bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %最好的染色体
avgfitness=sum(individuals.fitness)/sizepop; %染色体的平均适应度                              
trace=[avgfitness bestfitness]; % 记录每一代进化中最好的适应度和平均适应度

for num=1:maxgen

    % 选择  
    individuals=select(individuals,sizepop);   
    avgfitness=sum(individuals.fitness)/sizepop; 
    %交叉  
    individuals.chrom=Cross(pcross,lenchrom,individuals,sizepop,bound);  
    % 变异  
    individuals.chrom=Mutation(pmutation,lenchrom,individuals,sizepop,num,maxgen,bound);      
    
    % 计算适应度   
    for j=1:sizepop  
        x=individuals.chrom(j,:); %个体 
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);     
    end  
    %找到最小和最大适应度的染色体及它们在种群中的位置
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);

    % 最优更新：代替上一次进化中最好的染色体
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;

    % 记录每一代进化中最好的适应度和平均适应度
    avgfitness=sum(individuals.fitness)/sizepop;
    trace=[trace;avgfitness bestfitness]; %记录每一代进化中最好的适应度和平均适应度
    % FitRecord=[FitRecord;individuals.fitness];
    % 上述这行代码搭配 FitRecord = []; 具体作用不清楚
 end

% figure(1)
% [r c]=size(trace);
% plot([1:r]',trace(:,2),'b--');
% title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
% xlabel('进化代数');ylabel('适应度');
% legend('平均适应度','最佳适应度');
% disp('适应度                   变量');

%% 将最优初始阈值权值赋予给网络预测
x=bestchrom;
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=reshape(B2,outputnum,1);

%% BP网络训练
net.trainParam.epochs = 1000;
net.trainParam.lr = 0.01;
net.trainParam.goal = 1e-6;

net.trainParam.max_fail = 6;
net.trainParam.showWindow = true;

net.divideParam.trainRatio = 0.85;   %训练集占比
net.divideParam.valRatio = 0.15;      %验证集占比
net.divideParam.testRatio = 0;     %测试集占比

% 训练网络
[net, tr] = train(net, inputn, outputn);

%% 预测结果
% predictions_trainData_Z = sim(net, trainData_Z(:, 1:4)');
predictions_testData_Z = sim(net, testData_input_Z(:, 1:4)');
 
%反归一化
% predictions_trainData_Z = predictions_trainData_Z' * std(trainData(:, 5)) + mean(trainData(:, 5));
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

toc
    
%% 保存网络
save('GABP54.mat', 'net');
% load('BPNN1.mat');
% net_testData_Z = sim(net, testData_input_Z(:, 1:4)');
% net_testData_Z = net_testData_Z' * std(trainData(:, 5)) + mean(trainData(:, 5));