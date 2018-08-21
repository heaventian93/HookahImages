%% To predict the category of a new image

load test2.mat
newImage = '271.jpg';
img = readAndPreprocessImage(newImage);
featureLayer = 'fc7';
imageFeatures = activations(net, img, featureLayer);
imageFeatures = reshape(imageFeatures,[size(imageFeatures,3),1]);
% Make a prediction using the classifier
label = predict(classifier, imageFeatures')
figure,imshow(newImage),title (char(label))

% 
% predictedLabels = predict(classifier, testFeatures');
% confMat = confusionmat(testLabels, predictedLabels);
% confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
% mean(diag(confMat))*100
% figure,plotconfusion(testLabels, predictedLabels)