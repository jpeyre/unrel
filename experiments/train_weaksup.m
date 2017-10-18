function W = train_weaksup()

	% Train our weakly-supervised model

	opts = config();

	% Training
	opts.split = 'train';
	opts.supervision = 'weak';
	opts.dataset = 'vrd-dataset';
	opts.annotatedpairs = 'annotated';
	opts.candidatespairs = 'candidates';
	opts.featurestype = {'spatial', 'appearance'}; % Train with [S+A] features
	opts.num_negatives = 0;
	%opts.num_negatives = 150000; % uncomment to train with additional negatives

	mess = sprintf('Training with %s supervision', opts.supervision);
	if opts.num_negatives > 0
		mess = sprintf('%s with additional %s negatives', mess, opts.num_negatives);
	end
	fprintf(mess);

	W = train(opts);



end

