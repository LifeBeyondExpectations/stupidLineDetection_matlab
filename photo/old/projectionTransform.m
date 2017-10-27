function [] = projectionTransform()


% img = flip(flip(imread('6.jpg'), 1), 2);
imgOrigin = imread('1.jpg');

theta = 0;

% tm = [cosd(theta) -sind(theta) shearX; ...
%     sind(theta) cosd(theta) shearY; ...
%     0 0 1];
tm = [1.4 0 0.0004; ...
      0.65 0.4 0.00005; ...
      0 0 1];
tform = projective2d(tm);

outputImage = imwarp(imgOrigin, tform);
figure
imshow(outputImage);
% imshow(imgOrigin)