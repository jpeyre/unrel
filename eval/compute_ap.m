function [ap, ub] = compute_ap(candidates, scores, gtpositives, overlap_thresh, IoUmode)

    % Sort the candidates by confidence scores
    [scores, indsort] = sort(scores, 'descend');
    candidates = Dataset.select(candidates, indsort);

    % Get the true positives candidates (IoU > thresh)
    [hit, labels]    = get_truepositives(candidates, gtpositives, overlap_thresh, IoUmode);

    % Compute AP
    num_positives = length(gtpositives.im_id);
    [~, ~, info]  = vl_pr(labels, scores, 'NumPositives', num_positives);
    ap            = info.ap;

    % Upperbound recall
    ub = sum(hit)/num_positives;

end

