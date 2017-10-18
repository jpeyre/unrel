function [W, res,pred] = linbcfwopt_online(params, blocks,sample_blocks,slack_blocks,bags,pairs)

% This is the same optimization method except that it does not store X and
% P as it is too big to fit in memory when N becomes large (>10^6)
% In this version, we load the features for each bag every time

rng(1);

n_iter 	 = params.n_iter;
nConst	 = params.nConst;
kapa	 = params.kapa;
n	     = params.n;
k	     = params.num_classes;
n_blocks = length(blocks);


% Init a Z in the convex hull
param.MSK_IPAR_LOG = 0;

T = zeros(n*k+nConst,1);
for i=1:n_blocks
    prob = create_prob( params, sample_blocks{i}, slack_blocks{i}, bags, pairs);
    prob.c = rand(length(blocks{i}),1);
    [~,res] = mosekopt('minimize echo(0)',prob,param);
    Ti = res.sol.itr.xx;
    T(blocks{i}) = Ti;
end


fprintf('Begin LINBCFW iterations ...\n');

% Init block gap and random numbers for sampling blocks
gap = 10*ones(length(blocks),1);
samples = rand(n_iter+1,1);

% Init W and correlation matrix C by block
Z = reshape(T(1:n*k),n,k);
fprintf('init W ...\n');
[ C, W ] = initCW_by_block( params, Z, sample_blocks, pairs );


display = 1000;
display_loss_value = 0;
gap_values = [];
epoch_update = 5;
gap_update = epoch_update*n_blocks;

for i=1:n_iter

    if mod(i,gap_update) == 0
        fprintf('updating gap block ...\n');
        
        Z = reshape(T(1:n*k),n,k);
        
        for j=1:n_blocks
            fprintf('block : %d\n',j);
            current_block = blocks{j};
            current_block_sample = sample_blocks{j};
            current_slack_block = slack_blocks{j};
            
            % Load block features
            Xj = load_blockfeats(current_block_sample, pairs, params);
                
            % Compute gradient
            grad = Z(current_block_sample,:) - Xj*W;
            grad = reshape(grad,[],1);

            % Add slacks to the gradient
            if length(current_slack_block) > 0
                slacks = T(n*k+1:end); 
                grad = cat(1,grad,n*kapa*slacks(current_slack_block));
			end
           
			% Linear oracle
            prob = create_prob(params, sample_blocks{j}, slack_blocks{j}, bags, pairs);
            prob.c = grad;
            [~,Tmin] = mosekopt('minimize echo(0)',prob);
            Tmin = Tmin.sol.itr.xx;

			% Gap update
            diff = T(current_block)-Tmin;
            gap(j) = diff'*grad;
            gap(j) = 1/n*gap(j);  
			
		end
		
        total_gap = sum(gap);
        gap_values = [gap_values; total_gap];
        fprintf('total gap value: %.3e\n',total_gap);
	end

    % Adaptatively sampling a block
    r = samples(i+1);
    if any(gap) <0
        gap
    end
    prob_distribution = [0;cumsum(gap(:))/sum(gap)];
    [~,j] = histc(r,prob_distribution); 

    current_block = blocks{j};
    current_block_sample = sample_blocks{j};
    current_slack_block = slack_blocks{j};

    if mod(i,display) == 0
        fprintf(sprintf('iteration: %02d \t',i));
    end

    Z = reshape(T(current_block(1:end-length(current_slack_block))),[],k);   
        
    % Load block feature
    Xj = load_blockfeats(current_block_sample, pairs, params);
    
    % Compute gradient
    grad = (Z - Xj*W); 
    grad = reshape(grad,[],1);

    % Add slacks to gradient
    if length(current_slack_block) > 0
        slacks = T(n*k+1:end); 
        grad = cat(1,grad,n*kapa*slacks(current_slack_block));
    end
   
    % Linear oracle
    prob = create_prob(params, sample_blocks{j}, slack_blocks{j}, bags, pairs);
    prob.c = grad;
    [~,Tmin] = mosekopt('minimize echo(0)',prob);
    Tmin = Tmin.sol.itr.xx;
    
    % Update gap
    diff = T(current_block)-Tmin;
    gap(j) = diff'*grad;
    gap(j) = 1/n*gap(j);
    diff_square = reshape(diff(1:end-length(current_slack_block)),[],k); 
    if length(current_slack_block) > 0
        d_slacks = diff(end+1-length(current_slack_block):end);
        n_slacks = kapa*norm(d_slacks)^2;
    else
        n_slacks = 0;
    end    
    aux = diff_square.*(diff_square-Xj*(C*Xj'*diff_square));  
    aux = 1/n*sum(aux(:)) + n_slacks;
   
    % Update gamma, W, T
    gamma_rate = min(1,gap(j)/aux);    
    W = W - gamma_rate*(C*Xj'*diff_square);  
    T(current_block) = T(current_block) -  gamma_rate*diff;
 
    
    if mod(i,display) == 0
        if display_loss_value
            T_current = T(current_block);
            T_square = reshape(T_current(1:end-length(current_slack_block)),[],k);
                         
            if length(current_slack_block) > 0
                d_slacks = T_current(end+1-length(current_slack_block):end);
                n_slacks = kapa*norm(d_slacks)^2;
            else
                n_slacks = 0;
            end

            block_loss = T_square.*(T_square-Xj*C*Xj'*T_square);
            block_loss = 1/n*sum(block_loss(:)) + n_slacks;

            fprintf('Block gap %d value: %.3e - gamma: %.3e - block loss: %.3e\n',j,gap(j),gamma_rate,block_loss);
       else            
           fprintf('Block gap %d value: %.3e - gamma: %.3e\n',j,gap(j),gamma_rate);
       end
    end
  
end

res = reshape(T(1:n*k),n,k);
pred = res;

end
