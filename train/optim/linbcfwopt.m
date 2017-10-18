function W = linbcfwopt(params,X,blocks,block_square,slack_block,prob)

	rng(1); 

	n_iter 	= params.n_iter;
	bias	= params.bias;
	lambda	= params.lambda;
	nConst	= params.nConst;
	kapa	= params.kapa;
	n	= params.n;
	k	= params.num_classes;


	% Add bias column at the end
	X = cat(2,X,bias*ones(size(X,1),1));


	fprintf('building P ...\n');
	P = build_P(X,lambda);


	fprintf('Initialize Z in the convex hull...\n');
	T = zeros(n*k+nConst,1);
	n_blocks = length(blocks);
	mosek_params.MSK_IPAR_LOG = 0;
	for i=1:n_blocks
		prob(i).c = rand(length(blocks{i}),1);
		[~,res] = mosekopt('minimize echo(0)',prob(i),mosek_params);
		Ti = res.sol.itr.xx;
		T(blocks{i}) = Ti;
	end


	fprintf('Begin LINBCFW iterations ...\n');

	gap = 10*ones(length(blocks),1); % init gap values
	samples = rand(n_iter+1,1);
	Z = reshape(T(1:n*k),n,k);

	fprintf('init W ...\n');
	W = P*Z;


	% Speed up : pre-compute the blocks to avoid column slicing each time 
	fprintf('building blocks ...\n');
	for i=1:n_blocks
		X_block{i} = X(block_square{i},:);
		P_block{i} = P(:,block_square{i});
	end
	clear X P

	display = 1000;
	display_loss_value = 0;
	gap_values = [];
	epoch_update = 5;
	gap_update = epoch_update*n_blocks;

	for i=1:n_iter

		% Update the gap for all blocks after certain number of iterations
		if mod(i,gap_update) == 0
			fprintf('updating gap block ...\n');

			Z = reshape(T(1:n*k),n,k);

			for j=1:n_blocks

				%fprintf('block : %d\n',j);
				current_block = blocks{j};
				current_block_square = block_square{j};
				current_slack_block = slack_block{j};

				% Block gradient computation 
				grad = Z(current_block_square,:) - X_block{j}*W; % from Z 
				grad = reshape(grad,[],1);

				if length(current_slack_block) > 0
					slacks = T(n*k+1:end); 
					grad = cat(1,grad,n*kapa*slacks(current_slack_block)); % add slacks
				end

				% Linear oracle computation
				prob(j).c = grad; 
				[~,Tmin] = mosekopt('minimize echo(0)',prob(j));
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

		% Sample block with gap sampling
		r = samples(i+1);
		if any(gap) <0
			gap
		end
		prob_distribution = [0;cumsum(gap(:))/sum(gap)];
		[~,j] = histc(r,prob_distribution);

		current_block = blocks{j};
		current_slack_block = slack_block{j};

		if mod(i,display) == 0
			fprintf(sprintf('iteration: %02d \t',i));
		end

		Z = reshape(T(current_block(1:end-length(current_slack_block))),[],k);   

		% Block gradient computation 
		grad = (Z - X_block{j}*W);
		grad = reshape(grad,[],1);

		if length(current_slack_block) > 0
			slacks = T(n*k+1:end); 
			grad = cat(1,grad,n*kapa*slacks(current_slack_block));
		end

		% Linear oracle computation
		prob(j).c = grad; 
		[~,Tmin] = mosekopt('minimize echo(0)',prob(j));
		Tmin = Tmin.sol.itr.xx;

		% Block-gap and line-search computation
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
		aux = diff_square.*(diff_square-X_block{j}*(P_block{j}*diff_square));   
		aux = 1/n*sum(aux(:)) + n_slacks;
		gamma_rate = min(1,gap(j)/aux);

		% W update
		W = W - gamma_rate*(P_block{j}*diff_square);  

		% Block update
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

				block_loss = T_square.*(T_square-X(current_block_square,:)*P(:,current_block_square)*T_square);   
				block_loss = 1/n*sum(block_loss(:)) + n_slacks;

				fprintf('Block gap %d value: %.3e - gamma: %.3e - block loss: %.3e\n',j,gap(j),gamma_rate,block_loss);
			else            
			   fprintf('Block gap %d value: %.3e - gamma: %.3e\n',j,gap(j),gamma_rate);
			end
		end

	end

	%pred = reshape(T(1:n*k),n,k);


end
