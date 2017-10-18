function [pairs, scores, annotations] = predict(W, opts)

    % Test time : make predictions for different setups with our model
    % Input : W the classifiers learned for the predicates
    % Output : candidate pairs of boxes, their scores and full annotations

    featurestype     = opts.featurestype;
    bias             = opts.bias;
    alpha_predicates = opts.alpha_predicates;
    alpha_language   = opts.alpha_language;
    alpha_objects    = opts.alpha_objects;
    use_languagescores = opts.use_languagescores;
    use_objectscores = opts.use_objectscores;
    use_visualscores = opts.use_visualscores;
    candidatespairs  = opts.candidatespairs;
    language_model   = fullfile(opts.dataroot, opts.language_model);

    dataset = Dataset(opts);

    %% Load box-level annotations
    
    annotations = dataset.get_full_annotations();

    %% Load candidates pair of boxes
    
    pairs = dataset.load_candidates(candidatespairs);
    
    %% Compute scores for candidates
    scores = zeros(size(pairs,1), size(W,2));
    

    % Compute predicate scores
    if use_visualscores > 0
        fprintf('Computing predicate scores for %s...\n',candidatespairs);
        features = dataset.load_visualfeatures(candidatespairs, pairs, featurestype);

        if bias > 0
        	features  = cat(2,features, bias*ones(size(features,1),1));
        end

        predicate_scores = features*W;
        scores = scores + alpha_predicates*predicate_scores;
    end


    % Compute language scores
    if use_languagescores > 0
        fprintf('Computing language scores...\n');
        features_language = dataset.load_languagefeatures(pairs);
        WL = load(language_model);
        WL = WL.W;
        language_scores = features_language*WL;
        scores = scores + alpha_language*language_scores;
    end

    % Add object scores
    if use_objectscores > 0
        fprintf('Loading the object scores...\n');
        datafolder = fullfile(dataset.datapath, candidatespairs);
        object_scores = dataset.load_objectscores(datafolder, pairs);
        object_scores = object_scores*ones(1, size(scores,2));
        scores = scores + alpha_objects*object_scores;
    end

end
