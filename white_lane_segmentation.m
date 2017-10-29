function [] = white_lane_segmentation()

%%set mode
display = [1, 0, 1, 0, 0, 0, 0, 0];
% index == 1 : original file
% index == 2 : red component
% index == 3 : saturation component
% index == 4 : Opening component
% index == 5 : Opening -> Dila
% index == 6 : Threshold
% index == 7 : Labeling
% index == 8 : Visualized Labels

%%
threshold = 0
radius_of_structure_element = 4
se_opening = strel('disk', radius_of_structure_element);


%%
addrOfPhotos = dir(strcat(pwd, '/photo/*.jpg'));


for i = [12] %1: length(addrOfchitos)
    
    imgOrigin = imread(strcat(addrOfPhotos(i).folder, '/', addrOfPhotos(i).name));
    imgOrigin = imresize(imgOrigin, floor([size(imgOrigin, 1), size(imgOrigin, 2)] / 10));
    
    disp('size of the original image is')
    size(imgOrigin)
    
%     figure('name', 'origin', 'WindowStyle', 'Docked')
%     imshow(imgOrigin, 'InitialMagnification', 'fit')  
    
    %         imgOrigin = myThreshold(imgOrigin, 150);
    %         figure('name', 'threshold' , 'WindowStyle', 'Docked')
    %         imshow(imgOrigin, 'InitialMagnification', 'fit')
    
    %         imgOrigin = rgb_shift(imgOrigin, [3], 30);
    %         figure('name', 'rgb_shift', 'WindowStyle', 'Docked')
    %         imshow(imgOrigin, 'InitialMagnification', 'fit')
    
    % plot_rgb_image(imgOrigin);
    %img_hsv = plot_hsv_image(imgOrigin);
    img_hsv = rgb2hsv(imgOrigin);
    % max(img_hsv(:)) == 1
    
    for j = 3
        figure('name', ['hsv_is_', num2str(j)] , 'WindowStyle', 'Docked')
        imshow(img_hsv(:,:,j), 'InitialMagnification', 'fit')
        
        imgOpening = imopen(img_hsv(:,:,j), se_opening);
        figure('name', ['hsv_is_', num2str(j), '_Opening_radius_is_', num2str(radius_of_structure_element)] , 'WindowStyle', 'Docked')
        imshow(imgOpening, 'InitialMagnification', 'fit')
        
        figure('name', ['subtract'] , 'WindowStyle', 'Docked')
        imgTmp = img_hsv(:,:,j) - imgOpening;
        imshow(imgTmp, 'InitialMagnification', 'fit')
        
        max(max(imgTmp))
        
        % threshold = 50
        for threshold = 10:10:60
            imgThreshold = imbinarize(imgTmp, (threshold / 255));
            %imgThreshold = myThreshold(imgTmp, threshold/255);
            figure('name', ['hsv_is_', num2str(j), 'threshold_', num2str(threshold)], 'WindowStyle', 'Docked')
            imshow(imgThreshold, 'InitialMagnification', 'fit')
        end
    end
end