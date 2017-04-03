function [images, masks] = getBatchDeepmask(opts, imdb, batch)

    images = single(imdb.imdb(:,:,:,batch));
    masks = single(imdb.masks(:,:,1,batch));
    
    if numel(opts.gpus) > 0
        images = gpuArray(images);
        masks = gpuArray(masks);
    end
    
end