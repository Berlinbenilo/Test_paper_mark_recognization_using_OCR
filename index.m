close all
clear
clc
[filename,filepath]=uigetfile('file selector');
x=strcat([filepath,filename]);
originalImage = imread(x);
figure;
imshow(originalImage);
title('Original Image');
% Conversion to grayScale image
grayImage = rgb2gray(originalImage);
% Conversion to binary image
threshold = graythresh(grayImage);
binaryImage = ~im2bw(grayImage,0.5);
binaryImage=imclearborder(binaryImage);
% Removes all object containing fewer than 30 pixels
moddedImage = bwareaopen(binaryImage,1500);

pause(1)
% Showing binary image
figure(2);
imshow(moddedImage);
title('Modified Image');
% Labelling connected components
[L,Ne] = bwlabel(moddedImage);
% Measuring properties of image regions
propied = regionprops(L,'BoundingBox');
% g=ocr(orginalImage,propied.BoundingBox)
hold on
% Plot Bounding Box
for n=1:size(propied,1)
    rectangle('Position',propied(n).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off
pause (1)
%%
load matlab.mat
 %load test.mat
for n=1:Ne
    [r,c] = find(L==n);
    n1 = moddedImage(min(r):max(r),min(c):max(c));
    n1 = imresize(n1,[128 128]);
    n1 = imgaussfilt(double(n1),1);
    n1 = padarray(imresize(n1,[20 20],'bicubic'),[4 4],0,'both');
    imshow(~n1);
    fullFileName = fullfile('segmentedImages', sprintf('image%d.png', n));
    imwrite(n1, fullFileName);
    pause(1)
    x1=imread(fullFileName);
    ds= augmentedImageDatastore(imageSize,x1, 'ColorPreprocessing', 'gray2rgb');
    imagefeature = activations(net, ds, featureLayer, ...
        'MiniBatchSize', 32, 'OutputAs', 'columns');
    op(n,:) = predict(classifier,imagefeature, 'ObservationsIn', 'columns');
end
p=grp2idx(op)-1;
output=sum(p);
disp(output);