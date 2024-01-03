clc;
clear all;
close all;

tic

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
numIterations = 200;

% 输入层、隐藏层、输入层节点个数
inputnum = 4;
hiddennum = 7;
outputnum = 1;

% 初始化结果矩阵
results = zeros(size(dataset, 1), numIterations + 1);

%% Selfhelp
% 随机抽取1000次
for Num_selfhelp = 1:numIterations
	disp(['epoch = ', num2str(Num_selfhelp)]);
    trainIndices = datasample(1:size(dataset, 1), 60, 'Replace', false);
    testIndices = setdiff(1:size(dataset, 1), trainIndices);
    
    % 归一化处理
    trainData = dataset(trainIndices, :);
    testData = dataset(testIndices, :);
    
    trainData_Z = (trainData - mean(trainData) )./ std(trainData);
    
    % 测试集的归一化同样采用训练集的参数【防止数据泄露】
    testData_input_Z = (testData - mean(trainData) )./ std(trainData);
	
	inputn =  trainData_Z(:, 1:4)';
	outputn = trainData_Z(:, 5)';
    
    %% 创建神经网络
    % 输入数据为：4*60 + 1*60 + 隐含层神经元为X （连接权值NumWeightElements有4*X+X*1+X+1=6X+1）
    net = newff(inputn, outputn, hiddennum);
    
	%% 遗传算法参数初始化
	maxgen=50;                         %进化代数，即迭代次数
	sizepop=30;                        %种群规模
	pcross=0.3;                       %交叉概率选择，0和1之间
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

    %% 设置训练参数
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-6;
    net.trainParam.lr = 0.01;
    net.trainParam.max_fail = 12;
    net.trainParam.showWindow = false;

    net.divideParam.trainRatio = 0.8;   %训练集占比
    net.divideParam.valRatio = 0.2;      %验证集占比
    net.divideParam.testRatio = 0;     %测试集占比

    % 训练网络
    [net, tr] = train(net, inputn, outputn);
    
    %% 结果
    % 预测
    % predictions_trainData_Z = sim(net, trainData_Z(:, 1:4)');
    predictions_testData_Z = sim(net, testData_input_Z(:, 1:4)');
     
    %反归一化
    % predictions_trainData_Z = predictions_trainData_Z' * std(trainData(:, 5)) + mean(trainData(:, 5));
    predictions_testData_Z = predictions_testData_Z' * std(trainData(:, 5)) + mean(trainData(:, 5));
    
    for testData_size = 1:length(predictions_testData_Z)
        results(testIndices(testData_size), Num_selfhelp) = predictions_testData_Z(testData_size);
	end
end

%% 将非0值去除，组成全部由预测值的数组
% results中前1000列每一列都有13个数值，存放预测值，最后一列目前为空
[row, col] = size(results);
Results_nonZeroData = cell(row, 1);

for num_zero = 1:73
    nonZeroValues = results(num_zero, 1:col-1); % 取出每一行的除最后一列的数据
    nonZeroValues = nonZeroValues(nonZeroValues ~= 0); % 去除为零的值
    Results_nonZeroData{num_zero} = nonZeroValues; % 存储去除零后的数据到单元数组中
    results(num_zero, numIterations + 1) = mean(nonZeroValues); % 计算非零值的平均数
end

result_average = results(:, numIterations + 1);

toc

% % 写入excel，存在了sheet1中
% tableData = cell2table(Results_nonZeroData);
% filename = 'data0923.xlsx';
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
