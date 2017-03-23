function [im_] = scores_to_segmented_image( im )

    imageSize = size(im);
    im_ = zeros(imageSize(1), imageSize(2));
    
    for i = 1 : imageSize(1)
        for j = 1 : imageSize(2)
            [~,im_(i,j)] = max(im(i,j,:));
        end
    end

end

