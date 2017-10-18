function spatial = compute_spatial_features(pairs, gmm)

    % Compute raw spatial features
    spatial_features = extract_spatialconfig(pairs);
    
    % Quantized this representation
    spatial = apply_gmm(spatial_features, gmm.model, gmm.scaler); 
    
    % Append rel_id as first index
    spatial = [pairs.rel_id, spatial]; 

end

