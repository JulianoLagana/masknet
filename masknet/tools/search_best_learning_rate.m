learning_rates = logspace(-7,-4,10);
archs = {'masknet', 'masknet2', 'masknet3', 'deepmask_dag'};
imdb = 'VOC2012/pascal_imdb';

expDirs = cell(1,numel(learning_rates));

% For each architecture
for iArch = 1 : numel(archs)
    
    % Run 10 experiments with different learning rates
    % (each one of these takes around 5:30s to run using masknet)
    for iLr = 1 : numel(learning_rates)
        disp(['------------ ARCH: ' archs{iArch} ' | iLr: ' num2str(iLr) ' ----------']);
        run_experiment('imdb'         , imdb, ...
                       'arch'         , archs{iArch}             , ...
                       'learningRate' , learning_rates(iLr)     , ...
                       'weightDecay'  , 0                     , ...
                       'momentum'     , 0                     , ...
                       'batchSize'    , 40                    , ...
                       'numEpochs'    , 10                    , ...
                       'saveInterval' , 10                    , ...
                       'continue'     , true                   );

        expDirs{iLr} = [archs{iArch} '/' imdb '/lr' num2str(learning_rates(iLr)) '_wd0_mom0_batch40'];
    end
    
    % Plot results for the current architecture
    expDirs = strrep(expDirs,'.','p');
    compare_experiments(expDirs);
end




                         
                         
                        
                                        