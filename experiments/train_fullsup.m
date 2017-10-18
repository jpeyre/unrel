function W = train_fullsup()

    % Train our model with full supervision

    opts = config();
    opts.split = 'train';
    opts.supervision = 'full';
    opts.dataset = 'vrd-dataset';
    opts.featurestype = {'spatial', 'appearance'}; % Train with [S+A] features
    opts.num_negatives = 0; 
    opts.annotatedpairs = 'annotated';
    opts.candidatespairs = 'candidates';
    %opts.num_negatives = 150000; % uncomment to add negatives at training

    W = train(opts);


end

