function W = train_ridgereg(X,Y,params)     

	if params.bias>0
		X  = cat(2,X,params.bias*ones(size(X,1),1)); %add bias column at the end
	end

	n  = size(X,1);
	d  = size(X,2);

	Id = eye(d);
	B  = X.' * X + n * params.lambda * Id;
	B  = inv(B);
	W  = B * X.'*Y;

end

