function [ X ] = load_blockfeats( current_block_sample, pairs, params )

	% Load features for block i

	bias = params.bias;
	dataset = params.dataset;
	candidatespairs = params.candidates;
	featurestype = params.featurestype;

	% Get pairs in the block
	pairs_block = dataset.select(pairs, current_block_sample);

	% Load visual features
	X = dataset.load_visualfeatures(candidatespairs, pairs_block, featurestype );
	X = cat(2,X,bias*ones(size(X,1),1)); %add bias column at the end


end

