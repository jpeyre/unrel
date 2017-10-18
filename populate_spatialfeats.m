% Compute spatial features
% This step might take 10 minutes

dataroot = './data';
datasets = {'vrd-dataset','unrel-dataset'};
supervision = {'weak','full'};


% Load pre-computed GMM
gmm_models = struct();
for k=1:length(supervision)
    load(fullfile(dataroot, 'models', ['gmm-', supervision{k} ,'.mat']));
    gmm_models.(supervision{k}) = gmm;
end


for d=1:length(datasets)
    splits = dir(fullfile(dataroot, datasets{d}));
    splits(1:2) = [];
    splits(~[splits.isdir]) = [];

    for s=1:length(splits)
        candidates_dirs = dir(fullfile(dataroot, datasets{d}, splits(s).name));
        candidates_dirs(1:2) = [];
        candidates_dirs(~[candidates_dirs.isdir]) = [];

        for c=1:length(candidates_dirs)
            datapath = fullfile(dataroot, datasets{d}, splits(s).name, candidates_dirs(c).name);

            if ~exist(fullfile(datapath, 'features'))
                continue
            end

            fprintf('Compute spatial features for dataset:%s, split:%s, candidates:%s\n', datasets{d}, splits(s).name, candidates_dirs(c).name);

            % Get the pairs of boxes
            load(fullfile(datapath, 'pairs.mat'));

            for k=1:length(supervision)

                % Compute quantized spatial features using GMM
                all_spatial = compute_spatial_features(pairs, gmm_models.(supervision{k}));
                
                % Create spatial features directory if not exist
                if ~exist(fullfile(datapath, 'features', ['spatial-', supervision{k}]))
                    mkdir(fullfile(datapath, 'features', ['spatial-', supervision{k}]));
                end

                % Save features by image
                Images = unique(pairs.im_id);
                for i=1:length(Images)
                    im_id = Images(i);
                    indim = find(pairs.im_id==im_id);
                    spatial = all_spatial(indim,:);
                    save(fullfile(datapath, 'features', ['spatial', '-', supervision{k}], [num2str(im_id), '.mat']), 'spatial');
                end

            end

        end
    end

end


