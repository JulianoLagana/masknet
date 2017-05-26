function [ output_args ] = generate_pr_curves(  )
%GENERATE_PR_CURVES Summary of this function goes here
%   Detailed explanation goes here

    cat_names = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    load 'data/MAP scores/model topVal5000clevels.mat';    
    precision = truePositives./(truePositives+falsePositives);
    recall = truePositives./(truePositives+falseNegatives);
    for class = 1 : 20
        classPR_mask{class} = [recall(:,class) precision(:,class)];
    end
    
    load 'data/MAP scores/model noPMasks.mat';    
    precision = truePositives./(truePositives+falsePositives);
    recall = truePositives./(truePositives+falseNegatives);
    for class = 1 : 20
        classPR_nomask{class} = [recall(:,class) precision(:,class)];
    end
    
    selectedClasses = [12 8 4 17 2 9];
    for i = selectedClasses
        figure('name',cat_names{i}); hold on;
        plot(classPR_mask{i}(:,1), classPR_mask{i}(:,2),'*-'); 
        plot(classPR_nomask{i}(:,1), classPR_nomask{i}(:,2),'*-r'); 
        legend 'With partial masks' 'No partial masks';
        grid
        axis([0 1 0 1])
        xlabel recall
        ylabel precision
    end
        
    
    
end

