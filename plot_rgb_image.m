function [] = plot_rgb_image(imgOrigin)
    figure('name', 'red component', 'WindowStyle', 'Docked')
    imshow(imgOrigin(:,:,1))
    
     figure('name', 'green component', 'WindowStyle', 'Docked')
    imshow(imgOrigin(:,:,2))
    
     figure('name', 'blue component', 'WindowStyle', 'Docked')
    imshow(imgOrigin(:,:,3))
end