function [features] = apply_pca(features, coeff, scaler)

  % Scale the features to 0 mean 
  features = features - scaler.mu;
  %features = features ./ scaler.std;

  features = features*coeff;


end
