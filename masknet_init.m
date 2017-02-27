function [ net ] = masknet_init(  )

    % Meta parameters
    net.meta.inputSize = [14 14 512] ;
    net.meta.trainOpts.learningRate = 0.001 ;
    net.meta.trainOpts.numEpochs = 20 ;
    net.meta.trainOpts.batchSize = 31 ;  

    f = 1/100;
    net.layers = {};
    
    % Conv + ReLU layers
    net.layers{end+1} = struct('type', 'conv',...
                               'weights', {{f*randn(1,1,512,512, 'single'), zeros(1, 512, 'single')}} , ...
                               'stride', 1, ...
                               'pad', 0) ;
    net.layers{end+1} = struct('type','relu');
    
    % Conv layer
    net.layers{end+1} = struct('type','conv',...
                               'weights', {{f*randn(14,14,512,512,'single'), zeros(1,512,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;
                           
    % Conv layer                           
    net.layers{end+1} = struct('type','conv',...
                               'weights', {{f*randn(1,1,512,56^2,'single'), zeros(1,56^2,'single')}}, ...
                               'stride', 1, ...
                               'pad', 0) ;  
                           
    % Reshape layer                           
    net.layers{end+1} = struct('type','reshape',...
                               'newDim', [56,56,1]);
                           
    % Bilinear upsampling layer
    outDim = [224,224];
    grid = single(create_meshgrid(outDim, net.meta.trainOpts.batchSize));
    net.layers{end+1} = struct('type','bilinear',...
                               'grid',grid);     
           
    % Logistic log-loss layer                           
    net.layers{end+1} = struct('type','loss','loss','logistic','class', []);                          
                        
    % Tidy the net
    net = vl_simplenn_tidy(net);

end

