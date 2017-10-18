function W = train_diffrac_online(pairs, annotations, params)

    %% Form the blocks of decomposable problems
    fprintf('Forming image blocks...\n');
    [bags, image_bags, image_blocks] = form_blocks_pairs(pairs, annotations);

    params.nConst = length(bags); % number of constraints (equality provided each image-level label has at least one candidate pair of boxes)
    params.n = length(pairs.im_id); 

    %% Create blocks
    for i=1:length(image_blocks)  
       	blocks{i} = get_blocks(image_blocks{i}, params.n, params.num_classes); 	    % Z variables
       	blocks{i} = cat(2, blocks{i}, (params.n*params.num_classes+image_bags{i})); % slack variables
    end


	%% Solve the optimization problem with Block-coordinate Frank-Wolfe
    fprintf('Optimize with block-coordinate Frank-Wolfe...\n');
    [W, ~, ~] = linbcfwopt_online(params, blocks, image_blocks, image_bags, bags, pairs);

end


