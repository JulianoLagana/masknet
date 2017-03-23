function [ feature_map ] = compute_vgg_feature_map( vgg_feature_computer, img )

    % Preprocess image
    im_ = single(img) ; % note: 255 range
    im_ = imresize(im_, vgg_feature_computer.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - vgg_feature_computer.meta.normalization.averageImage ;

    % Compute feature maps
    res = vl_simplenn(vgg_feature_computer, im_, [], [], 'conserveMemory', true);
    feature_map = res(end).x;
    

end

