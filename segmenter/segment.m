function [ segmented_image ] = segment( net, im )

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

