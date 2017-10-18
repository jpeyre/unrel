function [ candidates, groundtruth ] = format_testdata_recall(pairs, scores, annotations, opts)

% This function format our outputs to match evaluation code of Lu16

zeroshot = opts.zeroshot;
path_ind_zeroshot = fullfile(opts.dataroot, opts.dataset, 'test', 'annotated', 'ind_zeroshot.mat');

%% Format groundtruth annotations

% If zeroshot, only evaluate recall on unseen triplets
if zeroshot ==1
    load(path_ind_zeroshot);
    annotations = Dataset.select(annotations, ind_zeroshot);
end

images = Dataset.get_images(annotations);
num_images = length(images);

groundtruth = struct('triplet', cell(1,num_images),...
                     'sub_box', cell(1,num_images),...
                     'obj_box', cell(1,num_images)...
                    );

for i=1:num_images
    im_id                   = images(i);
    idx                     = Dataset.get_pairs_in_image(annotations, im_id);
    groundtruth(i).triplet  = [annotations.sub_cat(idx), annotations.rel_cat(idx), annotations.obj_cat(idx)];
    groundtruth(i).sub_box  = annotations.subject_box(idx,:);
    groundtruth(i).obj_box  = annotations.object_box(idx,:);
end

%% Format the candidates

candidates = struct('scores',  cell(1,num_images),...
                    'triplet', cell(1,num_images),...
                    'sub_box', cell(1,num_images),...
                    'obj_box', cell(1,num_images),...
                    'rel_id', cell(1,num_images),...
                    'im_id', cell(1,num_images)...
                   );

for i=1:num_images
    im_id                 = images(i);
    idx                   = Dataset.get_pairs_in_image(pairs, im_id);
    candidates(i).rel_id  = pairs.rel_id(idx);
    candidates(i).im_id   = pairs.im_id(idx);
    candidates(i).sub_box = pairs.subject_box(idx,:);
    candidates(i).obj_box = pairs.object_box(idx,:);

    % Get top 1 prediction for each pair of boxes (to match Lu16 evaluation
    % method)
    scores_pred     = zeros(length(idx),1);
    rel_pred        = zeros(length(idx),1);
    for k=1:length(idx)
        [scores_pred(k), rel_pred(k)] = max(scores(idx(k),:));
    end

    candidates(i).scores  = scores_pred;
    candidates(i).triplet = [pairs.sub_cat(idx), rel_pred, pairs.obj_cat(idx)];
end



end

