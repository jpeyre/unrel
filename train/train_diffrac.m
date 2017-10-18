function W = train_diffrac(features, pairs, annotations, params)

    %% Form the blocks of decomposable problems
    fprintf('Forming image blocks...\n');
    [bags,image_bags,image_blocks] = form_blocks_pairs(pairs, annotations);

    %% Build constraints imposed by image-level labels
    fprintf('Building linear constraints...\n');
    linear_constraint = build_bag_constraints(params, bags, pairs);
    clear bags; 

    params.nConst = sum(cat(1, linear_constraint.slack));
    params.n = length(pairs.im_id); 

    %% Build all the optimization sub-problems in variable prob 
    fprintf('Building optimization problems...\n');
    for i=1:length(image_blocks)  
       	blocks{i} = get_blocks(image_blocks{i}, params.n, params.num_classes); 	    % Z variables
       	blocks{i} = cat(2, blocks{i}, (params.n*params.num_classes+image_bags{i})); % slack variables
    end

    [prob2.a, prob2.blc, prob2.buc]     = get_mosek_A_fw(linear_constraint, params.n, params.num_classes);
    [prob2.blx,~]                       = get_mosek_lx_fw(params.n, params.num_classes, params.nConst,[]);

    n_blocks = length(blocks);
    for i=1:n_blocks
        [prob(i).a, prob(i).blc, prob(i).buc] = get_blockmosek_A(prob2, blocks{i});  % bounds on AX
        [prob(i).blx,~] = get_blockmosek_lx(prob2, blocks{i}); 			    % lower bounds on variables
    end

    %% Solve the optimization problem with Block-coordinate Frank-Wolfe
    fprintf('Optimizing with block-coordinate Frank-Wolfe...\n');
    W = linbcfwopt(params, features, blocks, image_blocks, image_bags, prob);

end


