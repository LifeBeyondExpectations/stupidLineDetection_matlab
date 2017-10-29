function [] = extractRedLine_ver04()

%set mode
display = [1, 0, 1, 0, 0, 0, 0, 0];
% index == 1 : original file
% index == 2 : red component
% index == 3 : saturation component
% index == 4 : Opening component
% index == 5 : Opening -> Dilate
% index == 6 : Threshold
% index == 7 : Labeling
% index == 8 : Visualized Labels

setSingleDisp = 0
setSubaxis = 0
setShowThemAll = 1

%flag
set_hsv_1_sat_2_Intensity_3 = 1

% Parameter imshow(flipdim(imread('6.jpg'), 1))
threshold = 80;

% Define the dimension of the subplot

indexSubplot = 1;
addrOfPhotos = dir(strcat(pwd, '/photo/*.jpg'));
maxRow = sum(display);
if setSingleDisp
    maxCol = 1;
else
    maxCol = size(addrOfPhotos, 1);
end

feature = []; numFeature = [];
imgMatchOrigin = []; imgMatchBinary = [];

% SURF
ptsOriginal = []; ptsDistorted = [];

% tform = affine2d([0.4 -1 0; 1 1.1 0; 0 0 1])
% J = imwarp(img,tform);
% figure
% subplot(1,2,1)
% imshow(J)
% subplot(1,2,2)
% imshow(imgOrigin)


for i = [5] %1: length(addrOfPhotos)
    % I hate struct in matlab.
    
    imgOrigin = imread(strcat(addrOfPhotos(i).folder, '/', addrOfPhotos(i).name));
    imgOrigin = rgb_shift(imgOrigin, 1, 200);
    % I do not know why the 6.jpg show the vertical inverse
    if i == 6
        imgOrigin = flip(imgOrigin, 1);
    end
    
    if setSingleDisp
        figure
    end
    
    % plot the original image
    if display(1)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 1 : original file', 'WindowStyle', 'Docked')
                hold on
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('1st')
            end
        end        
        
        imshow(imgOrigin, 'InitialMagnification', 'fit')        
        indexSubplot = indexSubplot + maxCol;
    end
    
    % Red plot
    if display(2)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 2 : red component', 'WindowStyle', 'Docked')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('2nd')
            end
        end
        imshow(imgOrigin(:,:,1))
        indexSubplot = indexSubplot + maxCol;
    end
    
    % Saturation plot
    img = rgb2hsv(imgOrigin);
    imgSatu = img(:, :, 2);
    if display(3)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', '% index == 3 : saturation component', 'WindowStyle', 'Docked')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('3rd')
            end
        end
        
        imshow(imgSatu);
        imshow(img(:,:,1))
        indexSubplot = indexSubplot + maxCol;
    end
    
    % Opening
    se = strel('disk', 7);
    imgOpening = imopen(imgSatu, se);
    if display(4)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 4 : Opening component ' , 'WindowStyle', 'Docked')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('4th')
            end
        end
        imshow(imgOpening)
        indexSubplot = indexSubplot + maxCol;
    end
    
    % Dilate
    se = strel('disk', 40);
    imgOpeningDilate = imdilate(imgOpening, se);
    
    if display(5)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 5 : Opening -> Dilate', 'WindowStyle', 'Docked')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('5th')
            end
        end
        imshow(imgOpeningDilate)
        indexSubplot = indexSubplot + maxCol;
    end
    
    % threshold
    % note that the max value of the image is 1.0
    imgThreshold = imbinarize(imgOpeningDilate, (threshold / 255));
    if display(6)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 6 : Threshold ', 'WindowStyle', 'Docked')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('6th')
                indexSubplot
            end
        end
        imshow(imgThreshold)
        indexSubplot = indexSubplot + maxCol;
    end
    
    % Connected Component Labeling
    % bwconncomp > bwlabel. it use less memory.
    % https://kr.mathworks.com/help/images/ref/bwconncomp.html
    
    % Since bwlabel does not count the background(value == 0),
    % we must add some bias.
    imgThreshold = ~imgThreshold;
    
    [imgLabel, numLabel] = bwlabel(imgThreshold, 4);
    if display(7)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 7 : Labeling ', 'WindowStyle', 'Docked')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('7th')
            end
        end
        imshow(imgLabel, [])
        indexSubplot = indexSubplot + maxCol;
    end
    
    % we do not want to label the backgrounds
    for row = 1 : size(imgLabel, 1)
        for col = 1 : size(imgLabel, 2)
            if imgLabel(row, col) <= 1
                imgLabel(row, col) = 0;
            else
                imgLabel(row, col) = imgLabel(row, col) - 1;
            end
        end
    end
    
    %disp('unique(imgLabel) is')
    %unique(imgLabel)
    
    % Visualize label
    if display(8)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 8 : Visualized Labels', 'WindowStyle', 'Docked')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('8th')
            end
        end
        vislabels(imgLabel)
        indexSubplot = indexSubplot + maxCol;
    end
    
    % indexSubplot cannot be bigger than maxIndex of the subplot
    if indexSubplot > (maxCol * maxRow)
        % Draw
        if setSingleDisp
            indexSubplot = 1;
        else
            indexSubplot = i + 1;
        end
    end
    
%     imgMatchOrigin = cat(4, imgMatchOrigin, rgb2gray(imgOrigin));
%     imgMatchBinary = cat(4, imgMatchBinary, imgThreshold);
%     if isempty(ptsOriginal)
%         ptsOriginal  = detectSURFFeatures(imgMatchBinary(:, :, :, 1));
%     elseif isempty(ptsDistorted)
%         ptsDistorted = detectSURFFeatures(imgMatchBinary(:, :, :, 2));
%     end
            %     [featureNew, featureDim] = detectSURFFeatures(imgOpeningDilate);
%     feature = [feature, featureNew];
%     numFeature = [numFeature; featureDim];
end

% save('variables');
% featuresOriginal = []; validPtsOriginal = [];
% [featuresOriginal,validPtsOriginal] = extractFeatures(imgMatchBinary(:,:,:,1), ptsOriginal);
% [featuresDistorted,validPtsDistorted] = extractFeatures(imgMatchBinary(:,:,:,2), ptsDistorted);
% 
% index_pairs = matchFeatures(featuresOriginal,featuresDistorted);
% matchedPtsOriginal  = validPtsOriginal(index_pairs(:,1));
% matchedPtsDistorted = validPtsDistorted(index_pairs(:,2));
% figure; 
% showMatchedFeatures(imgMatchOrigin(:,:,:,1),imgMatchOrigin(:,:,:,2), matchedPtsOriginal,matchedPtsDistorted);
% title('Matched SURF points,including outliers');

% indexPairs = matchFeatures(feature(:, 1 : numFeature(1)), feature(:, (numFeature(1) + 1) : numFeature(2)));
% matchedPoints1 = valid_points1(indexPairs(:,1),:);
% matchedPoints2 = valid_points2(indexPairs(:,2),:);
% showMatchedFeatures(imgMatch(:,:,:,1), imgMatch(:,:,:,2),matchedPoints1,matchedPoints2);
end



