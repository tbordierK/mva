% EXERCISE1: basic training and testing of a classifier

% setup MATLAB to use our software
setup ;
clear;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Load training data
%encoding = 'bovw' ;
%encoding = 'vlad' ;
%encoding = 'fv' ;

categories = cell(3,1);
categories{1} = 'motorbike';
categories{2} = 'aeroplane';
categories{3} = 'person';

encodings = cell(3,1);
encodings{1} = 'fc7' ;
encodings{2} = 'fc8' ;
encodings{3} = 'soft' ;

ap = zeros(3,3,2);

for i =1:3
    encoding = encodings{i}
    for k =1:3
        
        
        category = categories{k};
        
        pos = load(['data/' category '_train_' encoding '.mat']) ;
        neg = load(['data/background_train_' encoding '.mat']) ;

        pos = pos.pos;
        neg = neg.pos;

        names = {pos.names{:}, neg.names{:}};
        
        if encoding(1:3) == 'fc7'
            POS = pos.fc7; NEG = neg.fc7;
        elseif encoding(1:3) == 'fc8'
            POS = pos.fc8; NEG = neg.fc8;
        else
            POS = pos.softmax;NEG = neg.soft;
        end

        histograms = [POS.', NEG.'] ;

        labels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
        clear pos neg ;

        % Load testing data
        pos = load(['data/' category '_val_' encoding '.mat']) ;
        neg = load(['data/background_val_' encoding '.mat']) ;

        pos = pos.pos;
        neg = neg.pos;

        testNames = {pos.names{:}, neg.names{:}};

        if encoding(1:3) == 'fc7'
            POS = pos.fc7; NEG = neg.fc7;
        elseif encoding(1:3) == 'fc8'
            POS = pos.fc8; NEG = neg.fc8;
        else
            POS = pos.soft;NEG = neg.soft;
        end

        testHistograms = [POS.', NEG.'] ;
        testLabels = [ones(1,numel(pos.names)), - ones(1,numel(neg.names))] ;
        clear pos neg ;

        % For stage G: throw away part of the training data
        %fraction = .1 ;
        %fraction = .5 ;
        fraction = +inf ;

        sel = vl_colsubset(1:numel(labels), fraction, 'uniform') ;
        names = names(sel) ;
        histograms = histograms(:,sel) ;
        labels = labels(:,sel) ;
        clear sel ;

        % count how many images are there
        %fprintf('Number of training images: %d positive, %d negative\n', ...
        %        sum(labels > 0), sum(labels < 0)) ;
        %fprintf('Number of testing images: %d positive, %d negative\n', ...
        %       sum(testLabels > 0), sum(testLabels < 0)) ;

        % For Stage E: Vary the image representation
        %histograms = removeSpatialInformation(histograms) ;
        %testHistograms = removeSpatialInformation(testHistograms) ;

        % For Stage F: Vary the classifier (Hellinger kernel)
        % ** insert code here for the Hellinger kernel using  **
        % ** the training histograms and testHistograms       **

        % L2 normalize the histograms before running the linear SVM
        %histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
        %testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;

        % L1 normalize the histograms before running the linear SVM
        % histograms = bsxfun(@times, histograms, 1./(sum(abs(histograms),1))) ;
        % testHistograms = bsxfun(@times, testHistograms, 1./(sum(abs(testHistograms),1))) ;

        % Hellinger : L1 normalize the histograms before running applying sqrt and feeding to the linear SVM
        %histograms = bsxfun(@times, histograms, 1./(sum(abs(histograms),1))) ;
        %testHistograms = bsxfun(@times, testHistograms, 1./(sum(abs(testHistograms),1))) ;
        %histograms = sqrt(histograms);
        %testHistograms = sqrt(testHistograms);




        % --------------------------------------------------------------------
        % Stage B: Training a classifier
        % --------------------------------------------------------------------

        % Train the linear SVM. The SVM paramter C should be
        % cross-validated. Here for simplicity we pick a valute that works
        % well with all kernels.
        C = 10 ;
        [w, bias] = trainLinearSVM(histograms, labels, C) ;

        % Evaluate the scores on the training data
        scores = w' * histograms + bias ;

        % Visualize the ranked list of images
        %figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
        %displayRankedImageList(names, scores)  ;

        % Visualize the precision-recall curve
        figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
        vl_pr(labels, scores) ;
        print(['train' category '_' encoding],'-dpng')
        [drop,drop,info] = vl_pr(labels, scores) ;
        ap(i,k,1) = info.ap;
 

        % Visualize visual words by relevance on the first image
        %[~,best] = max(scores) ;
        %displayRelevantVisualWords(names{best},w)


        % --------------------------------------------------------------------
        % Stage C: Classify the test images and assess the performance
        % --------------------------------------------------------------------

        % Test the linear SVM
        testScores = w' * testHistograms + bias ;

        % Visualize the ranked list of images
        %figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
        %displayRankedImageList(testNames, testScores)  ;

        % Visualize visual words by relevance on the first image
        %[~,best] = max(testScores) ;
        %displayRelevantVisualWords(testNames{best},w)

        % Visualize the precision-recall curve
        figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
        vl_pr(testLabels, testScores) ;
        print(['test' category '_' encoding],'-dpng')

        % Print results
        [drop,drop,info] = vl_pr(testLabels, testScores) ;
        fprintf('Test AP: %.2f\n', info.auc) ;
        
        [drop,perm] = sort(testScores,'descend') ;
        fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;
        
        ap(i,k,2) = info.ap;
        
    end
end
