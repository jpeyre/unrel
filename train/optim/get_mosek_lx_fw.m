function [ lx,ux] = get_mosek_lx_fw( n, P, nXi,IC )

	lx1 = zeros(n*P, 1);
	lx2 = zeros(nXi, 1);
	ux = inf(n*P+nXi,1);
	IC = find(cat(1,IC,zeros(n*P+nXi,1)));
	ux(IC) = 0;
	lx = cat(1, lx1, lx2);
	
end
