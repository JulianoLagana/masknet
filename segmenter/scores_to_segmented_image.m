function [im_] = scores_to_segmented_image( im )
% scores_to_segmented_image(im) transforms the output of FCN-8s to a single
% channeled image, where the value of each pixel is the number of the class
% to which it most probably belongs. This does so by going through all 
% channels for each pixel and substituting all of them for the index of the 
% highest value found.
%
% Inputs:
% 
%   - im : Output from FCN-8s. WxHxD image, where each pixel in channel 'k'
%   represents the probability that that pixel belongs to class 'k'. W and
%   H are specified in the 'meta.normalization.imageSize' field inside
%   'net'.
%   
% Outputs:
%
%   - im_ : WxHx1 image, where the value of each pixel is the number of the
%   class to which it most probably belongs.
%

    imageSize = size(im);
    im_ = zeros(imageSize(1), imageSize(2));
    
    for i = 1 : imageSize(1)
        for j = 1 : imageSize(2)
            [~,im_(i,j)] = max(im(i,j,:));
        end
    end

end

