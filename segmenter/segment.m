function [ segmented_image ] = segment( net, im )
% segment(net, im) performs semantic segmentation on the image 'im' using
% the FCN-8s network specified in 'net'. Note that this always performs the
% computation in the GPU.
%
% Inputs:
%   
%   - net: FCN-8s network in the DagNN format used by MatConvNet. 
% 
%   - im : Input image.
%   
% Outputs:
%
%   - segmented image : WxHxD image, where each pixel in channel 'k'
%   represents the probability that that pixel belongs to class 'k'. W and
%   H are specified in the 'meta.normalization.imageSize' field inside
%   'net'.
%

    I = gpuArray(im);
    
    im_ = single(I) ; % note: 0-255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;
    
    net.move('gpu');
    net.eval({'data', im_}) ;
    res = net.vars(end).value;
    
    res = gather(res);
    segmented_image = scores_to_segmented_image(res);


end

