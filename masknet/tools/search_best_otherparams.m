learning_rates = [2.1544e-6 4.6416e-6 4.6416e-5 1e-5];
archs = {'masknet', 'masknet2', 'masknet3', 'deepmask_dag'};
imdb = 'VOC2012/pascal_imdb';

expDirs = cell(1,4);
momentum = [0 0 0.9 0.9];
wd = [0 5e-5 0 5e-5];

% Find best hyperparameters for each architecture
for iArch = 1 : numel(archs)
    
    for iHyper = 1 : numel(momentum)
        disp(['------------ ARCH: ' archs{iArch} ' | iHyper: ' num2str(iHyper) ' ----------']);
        run_experiment('imdb'         , imdb, ...
                       'arch'         , archs{iArch}              , ...
                       'learningRate' , learning_rates(iArch)     , ...
                       'weightDecay'  , wd(iHyper)                , ...
                       'momentum'     , momentum(iHyper)          , ...
                       'batchSize'    , 40                        , ...
                       'numEpochs'    , 20                        , ...
                       'saveInterval' , 20                        , ...
                       'continue'     , true                      );

        expDirs{iHyper} = [archs{iArch} '/' imdb '/lr' num2str(learning_rates(iArch)) '_wd' num2str(wd(iHyper)) '_mom' num2str(momentum(iHyper)) '_batch40'];
    end
    
    % Plot results
    expDirs = strrep(expDirs,'.','p');
    compare_experiments(expDirs);   
    
end









                         
                         
                        
                                        