% 06-06  尝试对超参数进行调整，及学习速率，梯度下降的方法进行改进
clc, clear, close all

%% 加载train_dataset
data_x_loaded = readNPY('F:\xinyang\noise\data_x.npy');
data_y_loaded = readNPY('F:\xinyang\noise\data_y.npy');
data_x_loaded = permute(data_x_loaded,[2 3 4 1]);  % 转换维度
size_data_x_loaded = size(data_x_loaded);
XTrain = data_x_loaded;
YTrain = double(data_y_loaded);

%% 加载Validation_dataset
data_x_test_loaded = readNPY('F:\樊明月\CNN_Mingyue_Learning\data_x_test.npy');
data_y_test_loaded = readNPY('F:\樊明月\CNN_Mingyue_Learning\data_y_test.npy');
XValidation =permute(data_x_test_loaded,[2 3 4 1]);
YValidation =double(data_y_test_loaded);

size_XTrain = size(XTrain);
size_YTrain = size(YTrain);
size_XValidation = size(XValidation);
size_YValidation = size(YValidation);
%%%%%%
% 
% close all;
% 
% figure;
% a = reshape(XTrain(:,:,1,100),1024,1)
% plot(1:length(a),a,'r')
% 
% figure;
% b = reshape(XValidation(:,:,1,100),1024,1)
% plot(1:length(a),b,'b')
% 
% 
% 
% close all;
% figure;
% a = reshape(XTrain(:,:,2,1),1024,1)
% plot(1:length(a),a,'r')
% 
% figure;
% b = reshape(XValidation(:,:,2,1),1024,1)
% plot(1:length(a),b,'b')

%%%%%%%%% data_generater 为什么我翻译过来的分布情况和原来的差这么多呢%%%%%%%


%% 可视化

% close all;
% figure;
% idx = randperm(10000,4);
% for i = 1:numel(idx)
%     subplot(2,2,i)    
%     imshow(XTrain(:,:,1,idx(i)))
%     drawnow
%     title('训练集频率轴')
% end
% 
% figure;
% idx = randperm(100,4);
% for i = 1:numel(idx)
%     subplot(2,2,i)    
%     imshow(XValidation(:,:,1,idx(i)))
%     drawnow
%     title('验证集频率轴')
% end
%  figure;
% for i = 1:numel(idx)
%     subplot(2,2,i)    
%     imshow(XTrain(:,:,2,idx(i)))
%     drawnow
%     title('训练集幅度轴')
% end
% 
% figure;
% idx = randperm(100,4);
% for i = 1:numel(idx)
%     subplot(2,2,i)    
%     imshow(XValidation(:,:,2,idx(i)))
%     drawnow
%     title('验证集幅度轴')
% end
%  


%% 网络架构
layers_my = [
    imageInputLayer([32 32 2],"Name","imageinput")
    % layer_1
    convolution2dLayer([3 3],48,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    eluLayer(1,"Name","layer_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Stride",[2 2])
    % layer_2
    convolution2dLayer([3 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    eluLayer(1,"Name","layer_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Stride",[2 2])
    % layer_3
    convolution2dLayer([3 3],128,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    eluLayer("Name","elu_3")
    % layer_4
    convolution2dLayer([3 3],256,"Name","conv_4","Padding","same")
    
%   flattenLayer
%   dropoutLayer(0.2)
%   fullyConnectedLayer(256)
    fullyConnectedLayer(256)
    fullyConnectedLayer(1)
    regressionLayer("Name","regressionoutput")
];



%% 训练网络
miniBatchSize  = 64;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',2e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...    #　也许不使用学习率递减
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation},...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);
%     'ValidationData',{XValidation,YValidation},
%  'ValidationFrequency',500, ...
%
net = trainNetwork(XTrain,YTrain,layers_my,options);

YPredicted = predict(net,XValidation);
YValidation = double(YValidation);
predictionError = YValidation - YPredicted;
error_net = abs(YPredicted-YValidation);

squares = predictionError.^2;
rmse = sqrt(mean(squares))

mean_error =  mean(error_net );
std_error  = std(error_net );

figure(1);
plot(1:100,YValidation,'r');
hold on;
plot(1:100,YPredicted,'b');

figure(2);
plot(1:100,error_net,'b');
mean(YTrain)
sqrt(mse(YValidation))
mean(YValidation)

YTrain_double = double(YTrain);

close all;

 
 
 
 
 
 
 
 
 
 
















