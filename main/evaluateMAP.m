function [ averagePrecision ] = evaluateMAP( tp, fp, fn )

    cat_names = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat', ...
    'chair','cow','diningtable','dog','horse','motorbike','person','pottedplant', ...
    'sheep','sofa','train','tvmonitor'};
    
    averagePrecision = zeros(1,numel(cat_names));
    
    precision = tp./(tp+fp);
    recall = tp./(tp+fn);
    
    
    for class = 1 : numel(cat_names)
        
        classPR = [recall(:,class) precision(:,class)];     
        
        % Find unique recall values to integrate (add 0 and 1)
        uniqueRecallValues = unique(recall(:,class));
        uniqueRecallValues = union(uniqueRecallValues,[0 1]);
        
        topPrecision = 0;
        for iRecall = numel(uniqueRecallValues) : -1 : 2
            currentRecall = uniqueRecallValues(iRecall);
            nextRecall = uniqueRecallValues(iRecall-1);
            currentPrecision = max(precision(recall(:,class) == currentRecall,class));
            if currentPrecision > topPrecision
                topPrecision = currentPrecision;
            end
            
            deltaIntegral = topPrecision*(currentRecall - nextRecall);
            if ~isnan(deltaIntegral)
                averagePrecision(class) = averagePrecision(class) + deltaIntegral;
            end
        end
        
        subplot(4,5,class);
        plot(classPR(:,1), classPR(:,2),'*-'); 
        title([cat_names{class} '  AP: ' num2str(averagePrecision(class))]);
        axis([0 1 0 1]);        
        
    end


end

