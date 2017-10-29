function [img] = rgb_shift(img, which_rgb_to_shift, shift)

% RGB type for index 1, 2, 3
% img = imresize(img, [floor(size(img, 1)/4), floor(size(img,2)/4)]);

% paramter
% shift = 150

% fig1 = figure;
% set(fig1,'WindowStyle', 'Docked');
% set(fig1, 'name', 'img_origin')
% imshow(img, 'InitialMagnification', 'fit')

%max(max(img)) == 1

for i = 1: size(img, 1)
    for j = 1: size(img, 2)
        for k = which_rgb_to_shift %: size(img, 3)
            if img(i,j,k) > 255 - shift
                img(i, j, k) = img(i, j, k) + (shift - 255);
            else
                img(i, j, k) = img(i, j, k) + shift;
            end
        end
    end
end

% fig2 = figure;
% set(fig2,'WindowStyle', 'Docked');
% set(fig2, 'name', 'img_rotation_transfer')
% imshow(img, 'InitialMagnification', 'fit')

% save('test')



