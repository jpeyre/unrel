function [recall] = test_recall_Lu_baseline()

    %% Lu16 baseline for evaluation with recall

    opts = config();
    opts.split = 'test';
    opts.Nre  = 50;
    opts.zeroshot  = 1;
    opts.use_visualscores = 1;
    opts.use_languagescores = 0;

    %% Predicate detection

    opts.use_objectscores = 0;
    opts.candidatespairs = 'annotated';

    [pairs, scores, annotations]    = predict_baseline(opts);
    [candidates, groundtruth]       = format_testdata_recall(pairs, scores, annotations, opts); 
    [recall.predicate, ~]           = top_recall_Relationship(opts.Nre, candidates, groundtruth); 

    fprintf('R@%d for Predicate Detection : %.1f\n', opts.Nre, 100*recall.predicate);


    %% Relationship and Phrase Detection : use candidate pairs

    opts.use_objectscores = 1;
    opts.candidatespairs = 'Lu-candidates';


    [pairs, scores, annotations] = predict_baseline(opts);
    [candidates, groundtruth]    = format_testdata_recall(pairs, scores, annotations, opts); 
    [recall.relationship, ~] 	 = top_recall_Relationship(opts.Nre, candidates, groundtruth);
    [recall.phrase, ~]       	 = top_recall_Phrase(opts.Nre, candidates, groundtruth);

    fprintf('R@%d for Phrase Detection : %.1f\n', opts.Nre, 100*recall.phrase);
    fprintf('R@%d for Relationship Detection : %.1f\n', opts.Nre, 100*recall.relationship);

end