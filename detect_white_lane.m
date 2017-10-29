function [] = detect_white_lane()

%set mode
display = [1, 1, 1, 0, 0, 0, 0, 0];
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

color_threshold = 150

for i = 11
    imgOrigin = imread(strcat(addrOfPhotos(i).folder, '/', addrOfPhotos(i).name));
    


% for i = 11
%     imgOrigin = imread(strcat(addrOfPhotos(i).folder, '/', addrOfPhotos(i).name));
%     
%     img_r = imgOrigin(:, :, 1);
%     img_g = imgOrigin(:, :, 2);
%     img_b = imgOrigin(:, :, 3);
%     
%     img_h = imgOrigin(:, :, 1);
%     img_s = imgOrigin(:, :, 2);
%     img_v = imgOrigin(:, :, 3);
%     % threshold
%     % note that the max value of the image is 1.0
%     img_r = imbinarize(img_r, (color_threshold / 255));
%     img_g = imbinarize(img_g, (color_threshold / 255));
%     img_b = imbinarize(img_b, (color_threshold / 255));
%     
%     fig1 = figure;
%     set(fig1,'WindowStyle', 'Docked');
%     set(fig1, 'name', 'img_r')
%     imshow(imgOrigin, 'InitialMagnification', 'fit')
%     
%     fig2 = figure;
%     set(fig2,'WindowStyle', 'Docked');
%     set(fig2, 'name', 'img_r')
%     imshow(img_r, 'InitialMagnification', 'fit')
% 
%     fig3 = figure;
%     set(fig3,'WindowStyle', 'Docked');
%     set(fig3, 'name', 'img_r')
%     imshow(img_g, 'InitialMagnification', 'fit')
%     
%     fig4 = figure;
%     set(fig4,'WindowStyle', 'Docked');
%     set(fig4, 'name', 'img_r')
%     imshow(img_b, 'InitialMagnification', 'fit')
%     
%     
%     % threshold
%     % note that the max value of the image is 1.0
%     img_r = imbinarize(img_r, (color_threshold / 255));
%     img_g = imbinarize(img_g, (color_threshold / 255));
%     img_b = imbinarize(img_b, (color_threshold / 255));
%     
%     fig1 = figure;
%     set(fig1,'WindowStyle', 'Docked');
%     set(fig1, 'name', 'img_r')
%     imshow(imgOrigin, 'InitialMagnification', 'fit')
%     
%     fig2 = figure;
%     set(fig2,'WindowStyle', 'Docked');
%     set(fig2, 'name', 'img_r')
%     imshow(img_r, 'InitialMagnification', 'fit')
% 
%     fig3 = figure;
%     set(fig3,'WindowStyle', 'Docked');
%     set(fig3, 'name', 'img_r')
%     imshow(img_g, 'InitialMagnification', 'fit')
%     
%     fig4 = figure;
%     set(fig4,'WindowStyle', 'Docked');
%     set(fig4, 'name', 'img_r')
%     imshow(img_b, 'InitialMagnification', 'fit')
%     
% end
