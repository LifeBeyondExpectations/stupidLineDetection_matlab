function [] = extractRedLine_ver04()

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


for i = [5] %1: length(addrOfPhotos)
    % I hate struct in matlab.
    
    imgOrigin = imread(strcat(addrOfPhotos(i).folder, '/', addrOfPhotos(i).name));
    
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
                figure('name', 'index == 1 : original file')
                hold on
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('1st')
            end
        end        
        
        imshow(imgOrigin)        
        indexSubplot = indexSubplot + maxCol;
    end
    
    % Red plot
    if display(2)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', 'index == 2 : red component')
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
    if flag
    imgSatu = img(:, :, 2);
    if display(3)
        if setSubaxis
            subaxis(maxRow, maxCol, indexSubplot)
        else
            if setShowThemAll
                figure('name', '% index == 3 : saturation component')
            else
                subplot(maxRow, maxCol, indexSubplot)
                title('3rd')
            end
        end
        imshow(imgSatu);
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
                figure('name', 'index == 4 : Opening component ')
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
                figure('name', 'index == 5 : Opening -> Dilate')
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
                figure('name', 'index == 6 : Threshold ')
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
                figure('name', 'index == 7 : Labeling ')
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
                figure('name', 'index == 8 : Visualized Labels')
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


%%
function vislabels(L)
%VISLABELS Visualize labels of connected components
%   VISLABELS is used to visualize the output of BWLABEL.
%
%   VISLABELS(L), where L is a label matrix returned by BWLABEL,
%   displays each object's label number on top of the object itself.
%
%   Note: VISLABELS requires the Image Processing Toolbox.
%
%   Example
%   -------
%       bw = imread('text.png');
%       L = bwlabel(bw);
%       vislabels(L)
%       axis([1 70 1 70])

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

% Form a grayscale image such that both the background and the
% object pixels are light shades of gray.  This is done so that the
% black text will be visible against both background and foreground
% pixels.

background_shade = 200;
foreground_shade = 240;
I = zeros(size(L), 'uint8');
I(L == 0) = background_shade;
I(L ~= 0) = foreground_shade;

% Display the image, fitting it to the size of the figure.
imageHandle = imshow(I, 'InitialMagnification', 'fit');

% Get the axes handle containing the image.  Use this handle in the
% remaining code instead of relying on gca.
axesHandle = ancestor(imageHandle, 'axes');

% Get the extrema points for each labeled object.
s = regionprops(L, 'Extrema');

% Superimpose the text label at the left-most top extremum location
% for each object.  Turn clipping on so that the text doesn't
% display past the edge of the image when zooming.
hold(axesHandle, 'on');
for k = 1:numel(s)
    e = s(k).Extrema;
    text(e(1,1), e(1,2), sprintf('%d', k), ...
        'Parent', axesHandle, ...
        'Clipping', 'on', ...
        'Color', 'b', ...
        'FontWeight', 'bold');
end
hold(axesHandle, 'off');

end

%%

function [feature, dim] = SIFT_jaesung(imgRGB)
row=256;
colum=256;
%img=imread('lenna.jpg');
img=imresize(imgRGB,[row,colum]);

if size(img, 3) == 3
    img=rgb2gray(img);
end

% img=histeq(img);
img=im2double(img);
origin=img;
% img=medfilt2(img);
toc
%% Scale-Space Extrema Detection
tic
% original sigma and the number of actave can be modified. the larger
% sigma0, the more quickly-smooth images
sigma0=sqrt(2);
octave=3;%6*sigma*k^(octave*level)<=min(m,n)/(2^(octave-2))
level=3;
D=cell(1,octave);
for i=1:octave
    D(i)=mat2cell(zeros(row*2^(2-i)+2,colum*2^(2-i)+2,level),row*2^(2-i)+2,colum*2^(2-i)+2,level);
end
% first image in first octave is created by interpolating the original one.
temp_img=kron(img,ones(2));
temp_img=padarray(temp_img,[1,1],'replicate');
%%%%(2)
%%%%subplot(1,2,1);
%%%%imshow(origin)
%create the DoG pyramid.
for i=1:octave
    temp_D=D{i};
    for j=1:level
        scale=sigma0*sqrt(2)^(1/level)^((i-1)*level+j);
        p=(level)*(i-1);
        %%%%(1);
        %%%%subplot(octave,level,p+j);
        f=fspecial('gaussian',[1,floor(6*scale)],scale);
        L1=temp_img;
        if(i==1&&j==1)
            L2=conv2(temp_img,f,'same');
            L2=conv2(L2,f','same');
            temp_D(:,:,j)=L2-L1;
            %%%%imshow(uint8(255 * mat2gray(temp_D(:,:,j))));
            L1=L2;
        else
            L2=conv2(temp_img,f,'same');
            L2=conv2(L2,f','same');
            temp_D(:,:,j)=L2-L1;
            L1=L2;
            if(j==level)
                temp_img=L1(2:end-1,2:end-1);
            end
            %%%%imshow(uint8(255 * mat2gray(temp_D(:,:,j))));
        end
    end
    D{i}=temp_D;
    temp_img=temp_img(1:2:end,1:2:end);
    temp_img=padarray(temp_img,[1,1],'both','replicate');
end
toc
%% Keypoint Localistaion
% search each pixel in the DoG map to find the extreme point
tic
interval=level-1;
number=0;
for i=2:octave+1
    number=number+(2^(i-octave)*colum)*(2*row)*interval;
end
extrema=zeros(1,4*number);
flag=1;
for i=1:octave
    [m,n,~]=size(D{i});
    m=m-2;
    n=n-2;
    volume=m*n/(4^(i-1));
    for k=2:interval
        for j=1:volume
            % starter=D{i}(x+1,y+1,k);
            x=ceil(j/n);
            y=mod(j-1,m)+1;
            sub=D{i}(x:x+2,y:y+2,k-1:k+1);
            large=max(max(max(sub)));
            little=min(min(min(sub)));
            if(large==D{i}(x+1,y+1,k))
                temp=[i,k,j,1];
                extrema(flag:(flag+3))=temp;
                flag=flag+4;
            end
            if(little==D{i}(x+1,y+1,k))
                temp=[i,k,j,-1];
                extrema(flag:(flag+3))=temp;
                flag=flag+4;
            end
        end
    end
end
idx= extrema==0;
extrema(idx)=[];
toc
[m,n]=size(img);
x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
ry=y./2.^(octave-1-extrema(1:4:end));
rx=x./2.^(octave-1-extrema(1:4:end));
%%%%(2)
%%%%subplot(1,2,2);
%%%%imshow(origin)
hold on
%%%%plot(ry,rx,'r+');
%% accurate keypoint localization
%eliminate the point with low contrast or poorly localised on an edge
% x:|,y:-- x is for vertial and y is for horizontal
% value comes from the paper.
tic
threshold=0.1;
r=10;
extr_volume=length(extrema)/4;
[m,n]=size(img);
secondorder_x=conv2([-1,1;-1,1],[-1,1;-1,1]);
secondorder_y=conv2([-1,-1;1,1],[-1,-1;1,1]);
for i=1:octave
    for j=1:level
        test=D{i}(:,:,j);
        temp=-1./conv2(test,secondorder_y,'same').*conv2(test,[-1,-1;1,1],'same');
        D{i}(:,:,j)=temp.*conv2(test',[-1,-1;1,1],'same')*0.5+test;
    end
end
for i=1:extr_volume
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    rx=x+1;
    ry=y+1;
    rz=extrema(4*(i-1)+2);
    z=D{extrema(4*(i-1)+1)}(rx,ry,rz);
    if(abs(z)<threshold)
        extrema(4*(i-1)+4)=0;
    end
end
idx=find(extrema==0);
idx=[idx,idx-1,idx-2,idx-3];
extrema(idx)=[];
extr_volume=length(extrema)/4;
x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
ry=y./2.^(octave-1-extrema(1:4:end));
rx=x./2.^(octave-1-extrema(1:4:end));
%%%%figure(2)
%%%%subplot(2,2,3);
%%%%imshow(origin)
hold on
%%%%plot(ry,rx,'g+');
for i=1:extr_volume
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    rx=x+1;
    ry=y+1;
    rz=extrema(4*(i-1)+2);
    Dxx=D{extrema(4*(i-1)+1)}(rx-1,ry,rz)+D{extrema(4*(i-1)+1)}(rx+1,ry,rz)-2*D{extrema(4*(i-1)+1)}(rx,ry,rz);
    Dyy=D{extrema(4*(i-1)+1)}(rx,ry-1,rz)+D{extrema(4*(i-1)+1)}(rx,ry+1,rz)-2*D{extrema(4*(i-1)+1)}(rx,ry,rz);
    Dxy=D{extrema(4*(i-1)+1)}(rx-1,ry-1,rz)+D{extrema(4*(i-1)+1)}(rx+1,ry+1,rz)-D{extrema(4*(i-1)+1)}(rx-1,ry+1,rz)-D{extrema(4*(i-1)+1)}(rx+1,ry-1,rz);
    deter=Dxx*Dyy-Dxy*Dxy;
    R=(Dxx+Dyy)/deter;
    R_threshold=(r+1)^2/r;
    if(deter<0||R>R_threshold)
        extrema(4*(i-1)+4)=0;
    end
    
end
idx=find(extrema==0);
idx=[idx,idx-1,idx-2,idx-3];
extrema(idx)=[];
extr_volume=length(extrema)/4;
x=floor((extrema(3:4:end)-1)./(n./(2.^(extrema(1:4:end)-2))))+1;
y=mod((extrema(3:4:end)-1),m./(2.^(extrema(1:4:end)-2)))+1;
ry=y./2.^(octave-1-extrema(1:4:end));
rx=x./2.^(octave-1-extrema(1:4:end));
%%%%figure(2)
%%%%subplot(2,2,4);
%%%%imshow(origin)
hold on
%%%%plot(ry,rx,'b+');
toc
%% Orientation Assignment(Multiple orientations assignment)
tic
kpori=zeros(1,36*extr_volume);
minor=zeros(1,36*extr_volume);
f=1;
flag=1;
for i=1:extr_volume
    %search in the certain scale
    scale=sigma0*sqrt(2)^(1/level)^((extrema(4*(i-1)+1)-1)*level+(extrema(4*(i-1)+2)));
    width=2*round(3*1.5*scale);
    count=1;
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    %make sure the point in the searchable area
    if(x>(width/2)&&y>(width/2)&&x<(m/2^(extrema(4*(i-1)+1)-2)-width/2-2)&&y<(n/2^(extrema(4*(i-1)+1)-2)-width/2-2))
        rx=x+1;
        ry=y+1;
        rz=extrema(4*(i-1)+2);
        reg_volume=width*width;%3? thereom
        % make weight matrix
        weight=fspecial('gaussian',width,1.5*scale);
        %calculate region pixels' magnitude and region orientation
        reg_mag=zeros(1,count);
        reg_theta=zeros(1,count);
        for l=(rx-width/2):(rx+width/2-1)
            for k=(ry-width/2):(ry+width/2-1)
                reg_mag(count)=sqrt((D{extrema(4*(i-1)+1)}(l+1,k,rz)-D{extrema(4*(i-1)+1)}(l-1,k,rz))^2+(D{extrema(4*(i-1)+1)}(l,k+1,rz)-D{extrema(4*(i-1)+1)}(l,k-1,rz))^2);
                reg_theta(count)=atan2((D{extrema(4*(i-1)+1)}(l,k+1,rz)-D{extrema(4*(i-1)+1)}(l,k-1,rz)),(D{extrema(4*(i-1)+1)}(l+1,k,rz)-D{extrema(4*(i-1)+1)}(l-1,k,rz)))*(180/pi);
                count=count+1;
            end
        end
        %make histogram
        mag_counts=zeros(1,36);
        for x=0:10:359
            mag_count=0;
            for j=1:reg_volume
                c1=-180+x;
                c2=-171+x;
                if(c1<0||c2<0)
                    if(abs(reg_theta(j))<abs(c1)&&abs(reg_theta(j))>=abs(c2))
                        mag_count=mag_count+reg_mag(j)*weight(ceil(j/width),mod(j-1,width)+1);
                    end
                else
                    if(abs(reg_theta(j)>abs(c1)&&abs(reg_theta(j)<=abs(c2))))
                        mag_count=mag_count+reg_mag(j)*weight(ceil(j/width),mod(j-1,width)+1);
                    end
                end
            end
            mag_counts(x/10+1)=mag_count;
        end
        % find the max histogram bar and the ones higher than 80% max
        [maxvm,~]=max(mag_counts);
        kori=find(mag_counts>=(0.8*maxvm));
        kori=(kori*10+(kori-1)*10)./2-180;
        kpori(f:(f+length(kori)-1))=kori;
        f=f+length(kori);
        temp_extrema=[extrema(4*(i-1)+1),extrema(4*(i-1)+2),extrema(4*(i-1)+3),extrema(4*(i-1)+4)];
        temp_extrema=padarray(temp_extrema,[0,length(temp_extrema)*(length(kori)-1)],'post','circular');
        long=length(temp_extrema);
        minor(flag:flag+long-1)=temp_extrema;
        flag=flag+long;
    end
end
idx= minor==0;
minor(idx)=[];
extrema=minor;
% delete unsearchable points and add minor orientation points
idx= kpori==0;
kpori(idx)=[];
extr_volume=length(extrema)/4;
toc
%% keypoint descriptor
tic
d=4;% In David G. Lowe experiment,divide the area into 4*4.
pixel=4;
feature=zeros(d*d*8,extr_volume);
for i=1:extr_volume
    descriptor=zeros(1,d*d*8);% feature dimension is 128=4*4*8;
    width=d*pixel;
    %x,y centeral point and prepare for location rotation
    x=floor((extrema(4*(i-1)+3)-1)/(n/(2^(extrema(4*(i-1)+1)-2))))+1;
    y=mod((extrema(4*(i-1)+3)-1),m/(2^(extrema(4*(i-1)+1)-2)))+1;
    z=extrema(4*(i-1)+2);
    if((m/2^(extrema(4*(i-1)+1)-2)-pixel*d*sqrt(2)/2)>x&&x>(pixel*d/2*sqrt(2))&&(n/2^(extrema(4*(i-1)+1)-2)-pixel*d/2*sqrt(2))>y&&y>(pixel*d/2*sqrt(2)))
        sub_x=(x-d*pixel/2+1):(x+d*pixel/2);
        sub_y=(y-d*pixel/2+1):(y+d*pixel/2);
        sub=zeros(2,length(sub_x)*length(sub_y));
        j=1;
        for p=1:length(sub_x)
            for q=1:length(sub_y)
                sub(:,j)=[sub_x(p)-x;sub_y(q)-y];
                j=j+1;
            end
        end
        distort=[cos(pi*kpori(i)/180),-sin(pi*kpori(i)/180);sin(pi*kpori(i)/180),cos(pi*kpori(i)/180)];
        %accordinate after distort
        sub_dis=distort*sub;
        fix_sub=ceil(sub_dis);
        fix_sub=[fix_sub(1,:)+x;fix_sub(2,:)+y];
        patch=zeros(1,width*width);
        for p=1:length(fix_sub)
            patch(p)=D{extrema(4*(i-1)+1)}(fix_sub(1,p),fix_sub(2,p),z);
        end
        temp_D=(reshape(patch,[width,width]))';
        %create weight matrix.
        mag_sub=temp_D;
        temp_D=padarray(temp_D,[1,1],'replicate','both');
        weight=fspecial('gaussian',width,width/1.5);
        mag_sub=weight.*mag_sub;
        theta_sub=atan((temp_D(2:end-1,3:1:end)-temp_D(2:end-1,1:1:end-2))./(temp_D(3:1:end,2:1:end-1)-temp_D(1:1:end-2,2:1:end-1)))*(180/pi);
        % create orientation histogram
        for area=1:d*d
            cover=pixel*pixel;
            ori=zeros(1,cover);
            magcounts=zeros(1,8);
            for angle=0:45:359
                magcount=0;
                for p=1:cover;
                    x=(floor((p-1)/pixel)+1)+pixel*floor((area-1)/d);
                    y=mod(p-1,pixel)+1+pixel*(mod(area-1,d));
                    c1=-180+angle;
                    c2=-180+45+angle;
                    if(c1<0||c2<0)
                        if (abs(theta_sub(x,y))<abs(c1)&&abs(theta_sub(x,y))>=abs(c2))
                            
                            ori(p)=(c1+c2)/2;
                            magcount=magcount+mag_sub(x,y);
                        end
                    else
                        if(abs(theta_sub(x,y))>abs(c1)&&abs(theta_sub(x,y))<=abs(c2))
                            ori(p)=(c1+c2)/2;
                            magcount=magcount+mag_sub(x,y);
                        end
                    end
                end
                magcounts(angle/45+1)=magcount;
            end
            descriptor((area-1)*8+1:area*8)=magcounts;
        end
        descriptor=normr(descriptor);
        % cap 0.2
        for j=1:numel(descriptor)
            if(abs(descriptor(j))>0.2)
                descriptor(j)=0.2;
            end
        end
        descriptor=normr(descriptor);
    else
        continue;
    end
    feature(:,i)=descriptor';
end
index=find(sum(feature));
feature=feature(:,index);
dim = size(feature, 2);
end



