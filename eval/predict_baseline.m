function [pairs, scores, annotations] = predict_baseline(opts)

    % Test time : make predictions for different setups with our model
    % Input : W the classifiers learned for the predicates
    % Output : candidate pairs of boxes, their scores and full annotations

    use_languagescores = opts.use_languagescores;
    use_objectscores   = opts.use_objectscores;
    use_visualscores   = opts.use_visualscores;
    candidatespairs    = opts.candidatespairs;
    language_model     = fullfile(opts.dataroot, opts.language_model);

    dataset = Dataset(opts);

    % Load annotations
    annotations = dataset.get_full_annotations();

    % Load candidates
    pairs = dataset.load_candidates(candidatespairs);
    
    % Compute scores for candidates
    scores = ones(size(pairs,1), 70);
    

    % Compute predicate scores
    if use_visualscores > 0
        fprintf('Computing predicate scores...\n');
        visual_scores = load(fullfile(dataset.datapath, candidatespairs, 'predicate_scores','Lu-visual.mat'));
        visual_scores = visual_scores.scores;
        visual_scores = visual_scores(:,2:end);
        %scores       = scores.*visual_scores;
        scores        = max(visual_scores,1); %as done in Lu16 code
    end


    % Compute language scores
    if use_languagescores > 0
        fprintf('Computing language scores...\n');
        features_language = dataset.load_languagefeatures(pairs);
        WL = load(language_model);
        WL = WL.W;
        language_scores = features_language*WL;
        scores = scores.*language_scores;

    end

    % Add object scores
    if use_objectscores > 0
        fprintf('Loading the object scores...\n');
        objectscores = load(fullfile(dataset.datapath, candidatespairs, 'objectscores_Lu.mat'));
        objectscores = objectscores.object_scores;
        objectscores = objectscores*ones(1, size(scores,2));
        scores = scores.*objectscores;

    end

end
