function [ feature_map ] = compute_vgg_feature_map( vgg_feature_computer, img )

    net = vgg_feature_computer;

    % Preprocess image
    im_ = single(img) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = im_ - net.meta.normalization.averageImage ;

    % Compute feature maps
    res = vl_simplenn(net, im_);
    feature_map = res(end).x;
    

end

