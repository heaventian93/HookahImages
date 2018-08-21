%% Hooka classfication
% Extract the features using CNN and classify the images using SVM
clear;
clc;
close all;
%% Load the training images
outputFolder=pwd;
rootFolder = fullfile(outputFolder, 'data');
categories = {'hooka', 'non-hooka'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
tbl = countEachLabel(imds)
minSetCount = min(tbl{:,2}); 
imds = splitEachLabel(imds, minSetCount, 'randomize');
% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%% Load the pre-training network
net = alexnet()

%% Pre-process the Images
% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
[trainingSet, testSet] = splitEachLabel(imds, 0.2, 'randomize');
% Get the network weights 
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
w1 = imresize(w1,5);
figure,
montage(w1)
title('First convolutional layer weights')

%% Extract Training Features Using CNN
featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Plot the hist of the trainingFeatures
figure,bar(trainingFeatures)
xlabel('The training image features vector', 'FontSize', 10);ylabel('The range of the features', 'FontSize', 20); title('The features of the total training images', 'FontSize', 20)
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier 
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% Calculate the accuracy of the test images
% Extract test features using the CNN
testFeatures = activations(net, testSet, featureLayer, 'MiniBatchSize',32);
testFeatures = reshape(testFeatures,[size(testFeatures,3),size(testFeatures,4)]);
testLabels = testSet.Labels;

% Plot the hist of the testingFeatures
% figure,hist(testFeatures)
% xlabel('The testing image features vector', 'FontSize', 10);ylabel('The range of the features', 'FontSize', 20); title('The features of the total training images', 'FontSize', 20)

predictedLabels = predict(classifier, testFeatures');
confMat = confusionmat(testLabels, predictedLabels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
mean(diag(confMat))*100
figure,plotconfusion(testLabels, predictedLabels)



