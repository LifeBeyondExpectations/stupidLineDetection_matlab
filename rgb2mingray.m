function [img_min_gray] = rgb2mingray(img)
    img_min_gray = zeros(size(img, 1), size(img, 2), 'uint8');
    for i = 1: size(img, 1)
        for j = 1: size(img, 2)
            img_min_gray(i, j) = min(img(i, j, :));
        end
    end                
end