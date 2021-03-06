function [ img ] = createOverlayFromMultiImg( multiImg, varargin )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    opts.sensitivity = 1;
    opts.colors = distinguishable_colors(size(multiImg,3), [0 0 0]);
    opts = vl_argparse(opts,varargin);

    h = size(multiImg,1);
    w = size(multiImg,2);
    nInstances = size(multiImg,3);
    
    img = zeros(h,w,3);
    for i = 1 : nInstances
        c = opts.colors(i,:);
        img = img + opts.sensitivity*multiImg(:,:,i).*repmat(permute(c,[3 1 2]), h, w);
    end
    

end

