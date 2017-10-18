function [recall] = test_recall()

    opts = config();
    opts.split = 'test';
    
    % Training options
    opts.supervision = 'weak';
    opts.featurestype = {'spatial','appearance'};
    opts.num_negatives = 0;
    
    % Test options
    opts.use_languagescores     = 0; % set to 1 to use the language scores
    opts.Nre                    = 50; % k in recall@k
    opts.zeroshot               = 0; % set to 1 to evaluate on unseen triplets

    
    %% Load a pre-trained model (Nb. to train a model refer to
    % train_fullsup(), or train_weaksup() script)    
    
    model_name = [opts.supervision, '_', strjoin(opts.featurestype, '-'), '_bg', num2str(opts.num_negatives)];
    load(sprintf('%s/classifiers/%s', opts.dataroot, [model_name, '.mat']));

    
    %% Evaluate

    % Predicate Detection
    opts.candidatespairs     = 'annotated'; % use groundtruth pairs 
    opts.use_objectscores    = 0; % do not use object scores

    [pairs, scores, annotations] = predict(W, opts);
    [candidates, groundtruth]    = format_testdata_recall(pairs, scores, annotations, opts);
    [recall.predicate, ~]        = top_recall_Relationship(opts.Nre, candidates, groundtruth); % call evaluation code of [31]


    % Phrase/Relationship detection
    opts.candidatespairs    = 'Lu-candidates'; % use proposals of Lu16
    opts.use_objectscores   = 1; % use object scores

    [pairs, scores, annotations] = predict(W, opts);
    [candidates, groundtruth]    = format_testdata_recall(pairs, scores, annotations, opts);
    [recall.relationship, ~] 	 = top_recall_Relationship(opts.Nre, candidates, groundtruth);
    [recall.phrase, ~]       	 = top_recall_Phrase(opts.Nre, candidates, groundtruth);

    fprintf('R@%d for Predicate Detection : %.1f\n', opts.Nre, 100*recall.predicate);
    fprintf('R@%d for Phrase Detection : %.1f\n', opts.Nre, 100*recall.phrase);
    fprintf('R@%d for Relationship Detection : %.1f\n', opts.Nre, 100*recall.relationship);

end