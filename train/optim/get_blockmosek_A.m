function [a, blc, buc] = get_blockmosek_A(prob,block)

	A = prob.a(:,block);

	s = sum(abs(A),2);

	mask = find(s > 0);

	blc = prob.blc(mask);
	buc = prob.buc(mask);
	a = A(mask,:);

end
