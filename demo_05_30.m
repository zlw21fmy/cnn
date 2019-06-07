[trainImages,~,trainAngles] = digitTrain4DArrayData;
numTrainImages = size(trainImages,4);

figure
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)

    imshow(trainImages(:,:,:,idx(i)))
    drawnow
end

layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(12,25)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];