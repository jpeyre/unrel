function [ prob ] = create_prob( params, sample_block, sample_bag, bags, pairs)
% Create optimization problem related to movie block
% This replace the functions : get_blockmosek_lx, get_blockmosek_A,
% get_mosek_A_fw, get_mosek_lx_fw that store big matrices 
% As we have a lot of data, we do not want to store big matrices but rather
% create the problems online (even if it takes a bit longer...)

% current_block returns the current variables (both main and slack)
n = length(sample_block);
nXi = length(sample_bag);
P = params.num_classes;
dataset = params.dataset;

bags_block = bags(sample_bag);
pairs_block = dataset.select(pairs, sample_block);

% Get the linear constraints
linear_constraint = build_bag_constraints(params, bags_block, pairs_block);

% Define optimization problem
[ prob.a, prob.blc, prob.buc ] = get_mosek_A_fw( linear_constraint, n, P);
[ prob.blx,~] = get_mosek_lx_fw( n, P, nXi,[]);


end

