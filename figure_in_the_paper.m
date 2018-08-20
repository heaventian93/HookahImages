m=imread('100.jpg');

m1=imread('17.jpg');

figure,imshowpair(m1,m,'montage'),title('The examples of hookah and non-hookah images','FontSize', 15);

load hooka.mat

img=m1;
box=cell2mat(hooka{9,2});
detectedImg = insertObjectAnnotation(img,'rectangle',box,'hookah sign');
figure; imshow(detectedImg);

title('The hookah in the images', 'Fontsize',15)

load all_methods.mat

figure,plot(x,y,'r-o')
hold on 
plot(x,y1,'g--*')
hold on 
plot(x,y2,'b:+')
hold on 
plot(x,y3,'k-.x')
legend('CNN+SVM','CNN','SVM','BOF');
xlabel('Percentage of validated images (%)','FontSize', 15);
ylabel('Average Accuracy (%)','FontSize', 15);
title('Accuracy comparisons of different methods','FontSize', 15)