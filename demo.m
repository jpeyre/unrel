% This demo script allows you to train and evaluate our model in different setups. 
% To run this script, you would first need to change the paths accordingly in the init_paths.m
% You should have loaded and unzip the pre-processed data from the website
% Check that you run populate_spatialfeats.m that computes the spatial features
% Refer to config.m to change the training/test options

%% Setup %%

startup();
opts = config();


%% Training %%

opts.split         = 'train';
opts.supervision   = 'weak'; 
opts.num_negatives = 0;


W = train(opts);


%% Evaluation : recall on Visual Relationship Dataset %%

fprintf('Evaluate recall on Visual Relationship Detection\n');

opts.split = 'test';
opts.use_languagescores = 0; % set to 1 to use the language scores
opts.Nre                = 50; % k in recall@k
opts.zeroshot           = 0; % set to 1 to evaluate on unseen triplets


% Predicate Detection
opts.dataset          = 'vrd-dataset';
opts.candidatespairs  = 'annotated'; % use groundtruth pairs 
opts.use_objectscores = 0; % do not use object scores

[pairs, scores, annotations] = predict(W, opts);
[candidates, groundtruth]    = format_testdata_recall(pairs, scores, annotations, opts);
[recall.predicate, ~]        = top_recall_Relationship(opts.Nre, candidates, groundtruth); % call evaluation code of [31]


% Phrase/Relationship detection
opts.candidatespairs  = 'Lu-candidates'; % use proposals of Lu16
opts.use_objectscores = 1; % use object scores

[pairs, scores, annotations] = predict(W, opts);
[candidates, groundtruth]    = format_testdata_recall(pairs, scores, annotations, opts);
[recall.relationship, ~]     = top_recall_Relationship(opts.Nre, candidates, groundtruth);
[recall.phrase, ~]           = top_recall_Phrase(opts.Nre, candidates, groundtruth);
    
fprintf('R@%d for Predicate Detection : %.1f\n', opts.Nre, 100*recall.predicate);
fprintf('R@%d for Phrase Detection : %.1f\n', opts.Nre, 100*recall.phrase);
fprintf('R@%d for Relationship Detection : %.1f\n', opts.Nre, 100*recall.relationship);



%% Evaluation : retrieval of unusual relations on UnRel

fprintf('Evaluate retrieval on UnRel\n');

opts.split              = 'test';
opts.use_languagescores = 0;
opts.use_objectscores   = 0;
opts.IoUmode            = 'union'; % choose 'union', 'subject, 'subject-object'
opts.candidatespairs    = 'candidates'; opts.overlap = 0.3;
%opts.candidatespairs   = 'gt-candidates'; opts.overlap = 1'; opts.IoUmode= 'subject-object'; %uncomment to
% evaluate with GT


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

