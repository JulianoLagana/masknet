exps = {'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f100_M224', ...
        'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f200_M224', ...
        'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f300_M224', ...
        'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f100_M100', ...
        'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f300_M100', ...
        'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f512_M100', ...
        'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f100_M50', ...
        'deepmask_dag/partialcenteredimdb_lr2e-06_wd0_batch50'};
    
% exps = {'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f100_M224', ...
%         'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f200_M224', ...
%         'masknet3/partialcenteredimdb_lr2e-06_wd0_batch50_f300_M224', ...
%         'deepmask_dag/partialcenteredimdb_lr2e-06_wd0_batch50'};    
    
compare_experiments(exps);