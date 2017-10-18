function [ap, ub] = evaluate_retrieval(pairs, scores, annotations, opts)

    overlap_thresh  = opts.overlap;
    IoUmode         = opts.IoUmode; 
    load(fullfile(opts.dataroot, opts.vocab_objects));
    load(fullfile(opts.dataroot, opts.vocab_predicates));
    load(fullfile(opts.dataroot, opts.unusual_triplets));
    

    %% Compute AP for the annotated triplets
    ap = zeros(length(triplets),1);
    ub = zeros(length(triplets),1); % provide upperbound detection. Number of hits.

    for t=1:length(triplets)

        % Extract the groundtruth positives, the candidates and scores for the triplet query	
    	triplet = Dataset.get_triplet_index(triplets{t}, vocab_objects, vocab_predicates);
        
        % Get the positives annotations for this triplet
    	positives_triplet = Dataset.get_annotations_for_triplet(annotations, triplet); 
        
        % Get the candidates (matching subject-object)
        [candidates_triplet, idx] = Dataset.get_candidates_for_triplet(pairs, triplet); 
        
        % Get the scores (here : score of triplet = score of predicate)
        % because already filtered by object categories
        scores_triplet = scores(idx, triplet.rel);
        
        % scores_triplet = randperm(size(scores_triplet,1))';% random
        % ordering of the scores : uncomment to compute chance
        
        % Compute AP for this triplet
        [ap(t), ub(t)] = compute_ap(candidates_triplet, scores_triplet, positives_triplet, overlap_thresh, IoUmode);

        
    end

end

