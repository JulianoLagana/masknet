function [  ] = show_masknet_performance( net , imdb, start)

    % Pass the net to the gpu
    net.move('gpu') ;
    net.mode = 'test';
    net.conserveMemory = false;
    
    % Open imdb file
    file = matfile(imdb);
    nImages = getfield(whos(file),'size');  nImages = nImages(4);
    
    % Initialize fullscreen figure
    %figure('units','normalized','outerposition',[0 0 1 1]);
    
    for i = round(nImages*start)+1 : nImages
        
        % Load the image, the partial mask and the mask
        inputs = getBatch(file,i:i+49);
        
        % Process the feature map using the provided network
        net.eval(inputs);
        
        % Plot
        subplot(2,2,1);
        imgIdx = net.getVarIndex('input');
        im = gather(net.vars(imgIdx).value(:,:,:,1)/255);
        image(im);
        
        subplot(2,2,2);
        pmaskIdx = net.getVarIndex('pmask');
        pmask = gather(net.vars(pmaskIdx).value(:,:,:,1) > 0);
        image(imfuse(im, pmask ,'blend','Scaling','joint'));
        
        subplot(2,2,3);
        predIdx = net.getVarIndex('prediction');
        pred = gather(net.vars(predIdx).value(:,:,:,1) > 0);
        image(imfuse(im, pred ,'blend','Scaling','joint'));
        
        subplot(2,2,4);
        gtIdx = net.getVarIndex('gtMask');
        gt = gather(net.vars(gtIdx).value(:,:,:,1) > 0);
        image(imfuse(im, gt ,'blend','Scaling','joint'));
        
        % Wait for user visualization
        waitforbuttonpress;
        
    end
        

end

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch)
% --------------------------------------------------------------------
    images = single(imdb.imdb(:,:,:,batch));
    
    partial_masks = single(imdb.partial_masks(:,:,1,batch));
    partial_masks(partial_masks == 0) = -1;
    
    masks = single(imdb.masks(:,:,1,batch));
    masks(masks == 0) = -1;
    
    images = gpuArray(images);
    partial_masks = gpuArray(partial_masks);    
    masks = gpuArray(masks); 
    
    inputs = {'input',images,'pmask',partial_masks,'gtMask',masks};
end