function [ap] = test_retrieval()

    opts = config();
    opts.split = 'test';
    
    %% Load a pre-trained model (Nb. to train a model refer to
    % train_fullsup(), or train_weaksup() script)    
    opts.supervision = 'weak';
    opts.featurestype = {'spatial','appearance'};
    opts.num_negatives = 150000;
    
    model_name = [opts.supervision, '_', strjoin(opts.featurestype, '-'), '_bg', num2str(opts.num_negatives)];
    load(sprintf('%s/classifiers/%s', opts.dataroot, [model_name, '.mat']));


    %% Test options
    opts.candidatespairs = 'candidates'; opts.overlap = 0.3; opts.IoUmode = 'union'; % choose 'union', 'subject, 'subject-object'
    %opts.candidatespairs = 'gt-candidates'; opts.overlap = 1'; opts.IoUmode = 'subject-object'; %uncomment to evaluate with GT
    opts.use_languagescores = 0;
    opts.use_objectscores = 0;

    % Compute scores on UnRel
    opts.dataset = 'unrel-dataset';
    [unrel.pairs, unrel.scores, unrel.annotations] = predict(W, opts);

    % Compute scores on VRD 
    opts.dataset = 'vrd-dataset';
    [vrd.pairs, vrd.scores, vrd.annotations] = predict(W, opts);

    % Merge dataset
    [pairs, scores, annotations] = merge_datasets_for_retrieval(unrel, vrd); 

    % Compute AP
    [ap, ub] = evaluate_retrieval(pairs, scores, annotations, opts);
    fprintf('mAP=%.1f\n', 100*mean(ap));

end