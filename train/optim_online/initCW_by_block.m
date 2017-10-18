function [ C, W ] = initCW_by_block( params, Z, sample_blocks, pairs)

% Compute covariance matrix by block
% C = 0
% For each block :
% 1. Load block feature Xi 
% 2. C = C + Xi^T Xi
% 3. C = (C + M*lambda*Id)^-1
% Compute X^T*T by block as you do not want to load the features twice
% So the only thing we are storing are : T of size NxP, C of size dxd and W
% of size dxP. T could take a few GB to store but ok.

% Init size
Xi = load_blockfeats(sample_blocks{1}, pairs, params );
d = size(Xi,2);
P = size(Z,2);
lambda = params.lambda;

C = zeros(d,d);
M = 0; % store number of samples
prod_XT = zeros(d,P);

for i=1:length(sample_blocks)
    
    if mod(i,100)==0
        fprintf('Block features loaded %d / %d\n',i, length(sample_blocks));
    end
    
    % Load block feature
    Xi = load_blockfeats(sample_blocks{i}, pairs, params);
    
    % Compute cov matrix
    M = M + size(Xi,1);
    C = C + Xi' * Xi; 
    
    % Compute Xi^T*Ti
    prod_XT = prod_XT + Xi'*Z(sample_blocks{i},:);
    
end

C = inv(C + M*lambda*eye(size(C,1))); % regularize
W = C*prod_XT;

end

