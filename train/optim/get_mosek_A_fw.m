function [ A, blc, buc ] = get_mosek_A_fw( linear_constraint, n, P)

	nXi = sum(cat(1, linear_constraint.slack));
	nC = length(linear_constraint);

	A_subi = [];
	A_subj = [];
	A_subv = [];

	for i = 1:n
		A_subi = cat(1, A_subi, i*ones(P,1));
		A_subj = cat(1, A_subj, n*(0:(P-1))'+i);
		A_subv = cat(1, A_subv, ones(P,1));
	end

	blc = ones(n, 1);
	buc = ones(n, 1); 
	ti = cell(nC, 1);
	tj = cell(nC, 1);
	tv = cell(nC, 1);
	tl = cell(nC, 1);
	tu = cell(nC, 1);

	% Add the linear constraints
	for i = 1:nC
		sample = linear_constraint(i).sample;
		class = linear_constraint(i).class;
		slack = linear_constraint(i).slack;
		idx = find(kron(class, sample)); 
		ti{i} = (n+i) * ones(length(idx), 1);
		tj{i} = idx;
		tv{i} = ones(length(idx), 1);


		if slack~=0
			ti{i} = [ti{i}; n+i];
			tj{i} = [tj{i}; (n*P)+i];
			tv{i} = [tv{i}; slack];
		end

		value = linear_constraint(i).val;

		if strcmp(linear_constraint(i).type, 'geq')
			tl{i} = value;
			tu{i} = inf;
		elseif strcmp(linear_constraint(i).type, 'leq')
			tl{i} = -inf;
			tu{i} = value;
		else
			tl{i} = value;
			tu{i} = value;
		end
	end


	ti = cell2mat(ti);
	tj = cell2mat(tj);
	tv = cell2mat(tv);
	tl = cell2mat(tl);
	tu = cell2mat(tu);
	A_subi = cat(1, A_subi, ti);
	A_subj = cat(1, A_subj, tj);
	A_subv = cat(1, A_subv, tv);
	blc = cat(1, blc, tl);
	buc = cat(1, buc, tu); 

	A = sparse(A_subi, A_subj, A_subv, n+nC, n*P + nXi);


end


