function [blx,bux] = get_blockmosek_lx(prob,block)

	blx = prob.blx(block);

	bux = 0;

	if isfield(prob,'bux')
		bux = prob.bux(block);
	end

end
