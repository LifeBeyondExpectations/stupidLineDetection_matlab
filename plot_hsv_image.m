function [img] = plot_hsv_image(img)
    img = rgb2hsv(img);
    figure('name', 'hue', 'WindowStyle', 'Docked')
    imshow(img(:,:,1));
    figure('name', 'saturation', 'WindowStyle', 'Docked')
    imshow(img(:, :, 2));
    figure('name', 'Intensity', 'WindowStyle', 'Docked')
    imshow(img(:, :, 3));
end