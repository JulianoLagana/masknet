function [ ] = generate_pr_curve_comparison_pMasks( )


    cat_names = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};

    % Load the aggregators for the version with partial masks and compute
    % the precision-recall curve for each class
    load 'data/MAP scores/model topVal.mat';
    disp('AP for default:')
    AP_default = AP;
    disp(AP);
    precision = truePositives./(truePositives+falsePositives);
    recall = truePositives./(truePositives+falseNegatives);
    for class = 1 : 20
        classPR{class} = [recall(:,class) precision(:,class)];
    end
    
    % Do the same for the one using ground truth localization
    load 'data/MAP scores/model topVal - useGt loc.mat';
    disp('AP for perfect localization:')
    AP_loc = AP;
    disp(AP);
    precision = truePositives./(truePositives+falsePositives);
    recall = truePositives./(truePositives+falseNegatives);
    for class = 1 : 20
        classPR_loc{class} = [recall(:,class) precision(:,class)];
        
        % Add two more points to make it drawable
        classPR_loc{class}(1,:) = [-1 max(classPR_loc{class}(:,2))];
        classPR_loc{class}(end+1,:) = [max(classPR_loc{class}(:,1)) -1];
    end

    % Do the same for the one using ground truth masks
    load 'data/MAP scores/model topVal - useGt mask.mat';
    disp('AP for perfect masks:')
    AP_mask = AP;
    disp(AP);
    precision = truePositives./(truePositives+falsePositives);
    recall = truePositives./(truePositives+falseNegatives);
    for class = 1 : 20
        classPR_mask{class} = [recall(:,class) precision(:,class)];
    end
    
    % Do the same for the one using ground truth partial masks
    load 'data/MAP scores/model topVal - useGt pmask.mat';
    disp('AP for perfect pmasks:')
    disp(AP);    
    AP_pmask = AP;
    precision = truePositives./(truePositives+falsePositives);
    recall = truePositives./(truePositives+falseNegatives);
    for class = 1 : 20
        classPR_pmask{class} = [recall(:,class) precision(:,class)];
    end
    
    % Plot comparison of APs
    %plot(AP_default,'bo-','MarkerFaceColor','b');
    hold on;
    plot(AP_pmask-AP_default,'ro-','MarkerFaceColor','r');
    plot(AP_mask-AP_default,'mo-','MarkerFaceColor','m');
    plot(AP_loc-AP_default,'ko-','MarkerFaceColor','k');
    xticks(1:20);
    xticklabels(cat_names);
    xtickangle(-90);
    grid on;
    axis([1 20 -Inf 0.75]);
    legend 'Perfect semantic segmentation' 'Perfect masks' 'Perfect localization';    
    ylabel 'Absolute improvement';
    
    % Plot the resulting PR curves for some selected classes
    selectedClasses = [2 5 11 20];
    for i = selectedClasses
        figure('Name',cat_names{i});
        plot(classPR{i}(:,1), classPR{i}(:,2),'*-'); 
        hold on;
        plot(classPR_pmask{i}(:,1), classPR_pmask{i}(:,2),'*-'); 
        plot(classPR_mask{i}(:,1), classPR_mask{i}(:,2),'*-'); 
        plot(classPR_loc{i}(:,1), classPR_loc{i}(:,2),'*-'); 
        axis([0 1 0 1]);
        legend Default 'Perfect semantic segmentation' 'Perfect masks' 'Perfect Localization';     
        grid on;
        xlabel recall
        ylabel precision
    end
    

end

