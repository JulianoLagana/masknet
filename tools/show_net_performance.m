function [  ] = show_net_performance( net )

    % Remove the loss layer from the net
    net.layers(end) = [];

    % Pass the net to the gpu
    net = vl_simplenn_move(net, 'gpu') ;
    
    % Load VGG
    vgg = load_vgg_feature_computer('../data/imagenet-vgg-m.mat');
    
    % Open imdb file
    file = matfile('../data/imdb.mat');
    nImages = getfield(whos(file),'size');  nImages = nImages(4);
    
    % Initializa fullscreen figure
    figure('units','normalized','outerposition',[0 0 1 1]);
    
    for i = round(nImages*0.9)+1 : nImages
        
        % Load the image and the mask
        im = file.imdb(:,:,:,i);   
        im = gpuArray(im);
        mask = file.masks(:,:,:,i);
        
        % Compute the feature map using VGG
        fm = compute_vgg_feature_map(vgg,im);
        
        % Process the feature map using the provided network
        res = vl_simplenn(net,fm);
        
        % Plot the output overlaid on the orignal image
        im = gather(im);
        subplot(1,2,1);
        out = res(end).x;
        out = out(:,:,:,1);
        out = gather(out);
        image(imfuse(im,out>0,'blend','Scaling','joint'));
        axis image;
        
        % Plot the ground truth overlaid on the original image
        subplot(1,2,2);
        image(imfuse(im,mask,'blend','Scaling','joint'));
        axis image;
        
        % Wait for user visualization
        waitforbuttonpress;
        
    end
        

end

