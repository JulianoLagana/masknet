function [  ] = visualize_feature_maps( im, vgg )
    
    % Show image to user and wait for bounding boxes to compare
    image(im);
    bBox1 = getrect;
    bBox2 = getrect;
    
    % Compute entire feature map
    im = gpuArray(im);
    f_maps = compute_vgg_feature_map(vgg,im);
    
    % Resize feature map and crop to desired patches
    f_maps = imresize(f_maps,[size(im,1) size(im,2)]);
    f_maps = gather(f_maps);
    patch1 = cutPatch(f_maps, bBox1);
    patch2 = cutPatch(f_maps, bBox2);
    
    % Scale all dimensions of the feature map so that they're in 0-1 range
%     for i = 1 : size(patch1,3)
%        if any(any(patch1(:,:,i) ~= 0))
%           patch1(:,:,i) = patch1(:,:,i)-min(min(patch1(:,:,i)));
%           patch1(:,:,i) = patch1(:,:,i)/max(max(patch1(:,:,i))); 
%        end
%        if any(any(patch2(:,:,i) ~= 0))
%            patch1(:,:,i) = patch1(:,:,i)-min(min(patch1(:,:,i)));
%           patch2(:,:,i) = patch2(:,:,i)/max(max(patch2(:,:,i))); 
%        end
%     end
    
    % Concatenate them into image mosaic (only works if the patches are
    % h x w x 512)
    h1 = size(patch1,1);
    w1 = size(patch1,2); 
    h2 = size(patch2,1);
    w2 = size(patch2,2);
    nRows = 16;
    nColumns = 32;
    mosaic1 = zeros(h1*nRows,w1*nColumns);
    mosaic2 = zeros(h2*nRows,w2*nColumns);
    nImage = 1;
    for i = 0 : nRows -1
        for j = 0 : nColumns - 1
            mosaic1(h1*i+1 : h1*(1+i)  ,  w1*j+1 : w1*(1+j)) = patch1(:,:,nImage);
            mosaic2(h2*i+1 : h2*(1+i)  ,  w2*j+1 : w2*(1+j)) = patch2(:,:,nImage);
            nImage = nImage + 1;
        end
    end
    
    % Blend the mosaics for ease of comparison
    mosaic2 = imresize(mosaic2, [size(mosaic1,1) size(mosaic1,2)]);
    C = imfuse(mosaic1,mosaic2,'falsecolor','Scaling','joint');
    
    % Display the mosaics
    imshow(C);
    
    

end

function [ patch ] = cutPatch( im, bbox )

    bbox = round(bbox);
    x = bbox(1) + 1;
    y = bbox(2) + 1;
    w = bbox(3); % I don't really know why this is necessary, but it is...
    h = bbox(4);
    s = size(im);
    patch = im( y:min(y+h,s(1)) , x:min(x+w,s(2)) , : );

end
