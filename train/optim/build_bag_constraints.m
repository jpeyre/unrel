function linear_constraint = build_bag_constraints(params, bags, pairs)

	% Build linear constraints imposed by the image-level annotations

	num_bags = length(bags);
	N = length(pairs.im_id);


	A = pairs_in_bag(bags, pairs); % check which pairs is in which bag TO BE SPEED UP !


	eS = sparse(N, 1);
	eC = zeros(params.num_classes,1);

	linear_constraint = struct();
	count = 1;

	for i = 1:num_bags

		sidx = find(A(i, :)>0); % candidate pairs for bag i
		cidx = bags(i).rel_cat; % relation  

		if length(sidx) >= 0

			linear_constraint(count).sample       = eS;
			linear_constraint(count).sample(sidx) = 1;
			linear_constraint(count).class        = eC;
			linear_constraint(count).class(cidx)  = 1;
			linear_constraint(count).type         = 'geq';
			linear_constraint(count).val          = params.alpha;
			linear_constraint(count).slack        = 1;
			linear_constraint(count).weights      = ones(length(sidx),1);

			% If background class : constraint the rows of negatives to 1 instead
			if (params.bg > 0) & (cidx==1)
				linear_constraint(count).val      = 1*length(sidx);
			end

			count = count+1;
		end
	end


end
