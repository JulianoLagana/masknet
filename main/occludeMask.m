function mask = occludeMask( mask, bbox )
% occludeMask(mask, bbox) changes all pixel values of the binary mask
% 'mask' that lie inside the bounding box 'bbox' to zero.
%
% Inputs:
%
%   - mask : Binary image. Contains only one channel and all pixels belong
%   to {1,0}.
%   
%   - bbox : vector specifying the bounding box to be used. This vector is 
%   specified as [x y w h], where x and y are the coordinates of the top 
%   left corner of the bounding box and w and h specify the height and 
%   width of it. Note that the coordinate system used for x and y is 
%   1-based, meaning that the topmost left pixel of any image has
%   coordinates (1,1) (not 0,0).
%
% Outputs:
%
%   - mask : The transformed mask.
%

    maskSize = size(mask);

    bbox = round(bbox);
    x = max(bbox(1),1);
    y = max(bbox(2),1);
    w = bbox(3);
    h = bbox(4);
    
    maxx = min(x+w, maskSize(2));
    maxy = min(y+h, maskSize(1));
    
    mask(y:maxy, x:maxx) = 0;    

end

