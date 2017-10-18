function P = build_P(X,lambda)

	n = size(X,1);
	d = size(X,2);

	Id = eye(d);
	B = X.' * X + n * lambda * Id;
	B = inv(B);
	P = B * X.';


end

