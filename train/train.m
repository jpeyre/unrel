function W = train(opts)

	dataset         = Dataset(opts); % build dataset object
	featurestype    = opts.featurestype;
	num_negatives   = opts.num_negatives;


	switch opts.supervision
		case 'full'

			% Specify candidate pairs to use
			positivepairs = opts.annotatedpairs;  % positives are the annotated pairs
			negativepairs = opts.candidatespairs; % negatives are sampled from the proposals

			% Load features, annotations
			fprintf('Loading features and annotations for fully-supervised training...\n');
			[features, annotations, ~] = dataset.load_traindata_full(positivepairs, featurestype, num_negatives, negativepairs);

			% Train with ridge regression
			params_ridgereg.lambda	= opts.lambda;
			params_ridgereg.bias	= opts.bias;
			fprintf('Training ridge regression...\n');
			W = train_ridgereg(features, annotations, params_ridgereg);
			fprintf('Training done !\n');


		case 'weak'

			% Specify candidates : both positives and negatives are sampled from the proposals
			positivepairs = opts.candidatespairs;
			negativepairs = opts.candidatespairs;

			% Load features, annotations
			fprintf('Loading features and annotations for weakly-supervised training...\n');
			[features, annotations, pairs]  = dataset.load_traindata_weak(positivepairs, featurestype, num_negatives, negativepairs);

			if opts.num_negatives > 0
				  opts.num_predicates = opts.num_predicates + 1;
			end

			% Parameters for optimization
			params_diffrac.bias         	= opts.bias;
			params_diffrac.lambda           = opts.lambda;
			params_diffrac.num_classes      = opts.num_predicates;
			params_diffrac.n_iter           = opts.n_iter;
			params_diffrac.alpha            = opts.alpha;
			params_diffrac.kapa         	= opts.kapa;
			params_diffrac.bg           	= opts.num_negatives;


			% Call diffrac for training
			fprintf('Training with diffrac...\n');
			W = train_diffrac(features, pairs, annotations, params_diffrac);

			
			% Online training
			%params_diffrac.featurestype = opts.featurestype;
			%params_diffrac.candidates = opts.candidatespairs;
			%params_diffrac.dataset = dataset;
			%W = train_diffrac_online(pairs, annotations, params_diffrac);


	end


	if opts.num_negatives > 0
		W = W(:,2:end);
	end


end





