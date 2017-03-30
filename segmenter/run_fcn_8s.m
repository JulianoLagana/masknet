function [ segmented_image ] = run_fcn_8s( im, varargin )
% segment(net, im) performs semantic segmentation on the image 'im' using
% the FCN-8s network specified in 'net'. Note that this always performs the
% computation in the GPU.
%
% Inputs:
%   
%   - im : Cell array of input images.
%
%   - varargin : Cell array containing name value pairs for overriding the
%   default parameters.
%   
% Outputs:
%
%   - segmented image : Cell array of output images. Each cell contains a
%   WxHx1 image, where the value at each pixel is the number of the
%   class to which it most probably belongs according to the FCN-8s output.
%

    % Default parameters
    opts.gpu = 0;

    % Override default parameter with user supplied values
    opts = vl_argparse(opts,varargin);

    % Load FCN-8s
    net = dagnn.DagNN.loadobj(load('data/models/pascal-fcn8s-dag.mat'));
    if opts.gpu >= 1
        net.move('gpu');
    end

    for i = 1 : numel(im)
        
        % If necessary, move image to gpu
        if opts.gpu >= 1
            I = gpuArray(im{i});
        else
            I = im{i};
        end
        
        % Save size of the original image
        sz = size(I);
        sz = sz(1:2);
        
        % Pre-process image
        im_ = single(I) ; % note: 0-255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ; 
        
        % Run FCN-8s on the image
        net.eval({'data', im_}) ;
        res = net.vars(end).value;
        res = gather(res);    
        
        % Convert scores to segmentation
        segmented_image{i} = imresize(scores_to_segmented_image(res),sz,'nearest');

    end

end

