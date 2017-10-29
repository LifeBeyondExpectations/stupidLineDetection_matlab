function [img] = myThreshold(img, threshold)
    for i = 1: size(img, 1)
        for j = 1: size(img, 2)
            for k = 1:size(img, 3)
                if img(i,j,k) < threshold
                    for k =1: size(img, 3)
                        img(i,j,k)=0;
                    end
                    break;
                end
            end
        end
    end
end