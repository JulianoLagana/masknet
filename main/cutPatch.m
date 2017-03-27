function [ patch ] = cutPatch( im, bbox )
% cutPatch(im, bbox) returns the part of the image 'im' that the bounding
% box 'bbox' encloses.
%
% Inputs:
%
%   - im : Input image.
%   
%   - bbox : vector specifying the bounding box to be used to crop the image.
%   This vector is specified as [x y w h], where x and y are the coordinates
%   of the top left corner of the bounding box and w and h specify the
%   height and width of it. Note that the coordinate system used for x and
%   y is 1-based, meaning that the topmost left pixel of any image has
%   coordinates (1,1) (not 0,0).
%
% Outputs:
%
%   - patch : The cropped image.
%

    bbox = round(bbox);
    x = max(bbox(1), 1);
    y = max(bbox(2), 1);
    w = bbox(3);
    h = bbox(4);
    s = size(im);
    
    patch = im( y:min(y+h,s(1)) , x:min(x+w,s(2)) , : );

end

