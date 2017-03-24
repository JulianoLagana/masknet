function [ patch ] = cutPatch( im, bbox )

    bbox = round(bbox);
    x = max(bbox(1), 0) + 1;
    y = max(bbox(2), 0) + 1;
    w = bbox(3); % I don't really know why this is necessary, but it is...
    h = bbox(4);
    s = size(im);
    patch = im( y:min(y+h,s(1)) , x:min(x+w,s(2)) , : );

end

