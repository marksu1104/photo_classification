%Load Images
downloadFolder = 'C:\Users\marks\桌面\data\data';

% Get training images
flowerds = imageDatastore(downloadFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
% Split into training and testing sets
% Split the data set into a training and test data. Pick 80% of images from each set for the training data and the remainder 20% for the test data.
[trainImgs,testImgs] = splitEachLabel(flowerds,0.8);
% Determine the number of flower species
numClasses = numel(categories(flowerds.Labels));


% Load Pretrained Network %這裡改Network
net = alexnet;
layers = net.Layers;

% Modify the classification and output layers
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;

miniBatchSize = 40;
valFrequency = floor(numel(trainImgs.Files)/miniBatchSize);
% Set training algorithm options
% Lower the learning rate for transfer learning % 這裡加入training-progress的東西
options = trainingOptions('sgdm','MiniBatchSize',miniBatchSize,'InitialLearnRate', 0.001,'Shuffle','every-epoch','Verbose',false,'ValidationData',testImgs,'ValidationFrequency',valFrequency,MaxEpochs = 5,Plots="training-progress");


% Perform training
[flowernet,info] = trainNetwork(trainImgs, layers, options);


% Use the trained network to classify test images
[testpreds,scores] = classify(flowernet,testImgs);

%% Evaluate the results
% Calculate the accuracy
nnz(testpreds == testImgs.Labels)/numel(testpreds)
% Visualize the confusion matrix
cm = confusionchart(testImgs.Labels,testpreds,'RowSummary','row-normalized');
cm.Normalization = 'row-normalized'; 
sortClasses(cm,'descending-diagonal')
cm.Normalization = 'absolute'; 



%% Display Top Predictions
idx = find(testImgs.Labels == 'ME');
idx = idx(randperm(numel(idx),4));
testTop = testpreds(idx);


figure
for i = 1:4
    subplot(2,2,i);imshow(readimage(testImgs,idx(i)));title(string(testImgs.Labels(idx(i))) + ", " + num2str(100*max(scores(idx(i),:)),3) + "%");
end
