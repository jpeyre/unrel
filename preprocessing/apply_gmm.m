function [features] = apply_gmm(features, GMModel, scaler)

  % Scale the features to 0 mean and unit variance
  features = features - scaler.mu;
  features = features./scaler.std;

  % Get the posterior probabilities 
  features = posterior(GMModel, features);

end
