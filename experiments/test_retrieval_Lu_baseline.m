function [ap] = test_retrieval_Lu_baseline()

    %% Lu16 baseline for evaluation with retrieval

    opts = config();
    opts.split = 'test';
    opts.candidatespairs = 'candidates'; opts.overlap = 0.3; opts.IoUmode   = 'subject-object'; 
    opts.candidatespairs = 'gt-candidates'; opts.overlap = 1'; opts.IoUmode   = 'subject-object';

    % Path to pre-computed Lu16 scores
    path_scores = sprintf('%s/%s/%s/%s/%s/%s', opts.dataroot, '%s', 'test', opts.candidatespairs, 'predicate_scores', 'Lu-baseline.mat');


    % UNREL
    opts.dataset = 'unrel-dataset';
    dataset = Dataset(opts);
    % Load candidates/annotations
    unrel.annotations = dataset.get_full_annotations();
    unrel.pairs = dataset.load_candidates(opts.candidatespairs);
    % Load pre-computed scores
    load(sprintf(path_scores, opts.dataset));
    unrel.scores = scores(:,2:end); % first index is rel_id


    % VRD
    opts.dataset = 'vrd-dataset';
    dataset = Dataset(opts);
    % Load candidates/annotations
    vrd.annotations = dataset.get_full_annotations();
    vrd.pairs = dataset.load_candidates(opts.candidatespairs);
    % Load pre-computed scores
    load(sprintf(path_scores, opts.dataset));
    vrd.scores = scores(:,2:end); % first index is rel_id

    % Merge dataset
    [pairs, scores, annotations] = merge_datasets_for_retrieval(unrel, vrd); 

    % Compute AP
    [ap, ub] = evaluate_retrieval(pairs, scores, annotations, opts);
    fprintf('mAP=%.1f\n', 100*mean(ap));

end