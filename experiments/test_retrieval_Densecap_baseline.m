function [ap] = test_retrieval_Densecap_baseline()

    % DenseCap [1] provides only a region proposal (contrary to our method which provide a pair of boxes). 
    % Once can interpret this region proposal either as a subject box or union box. 
    % We obtained these candidate regions and scores by forwarding VRD and
    % UnRel dataset in the pre-trained DenseCap network released by the
    % authors. See : https://github.com/jcjohnson/densecap

    % [1] DenseCap: Fully Convolutional Localization Networks for Dense Captioning, Justin Johnson*, Andrej Karpathy*, Li Fei-Fei, 

    %% DenseCap baseline for evaluation with recall

    opts = config();
    opts.split = 'test';
    IoUmode = opts.IoUmode; 
    opts.overlap = 0.3; opts.IoUmode = 'subject'; 
    %opts.overlap = 1'; opts.IoUmode = 'subject-object';

    assert((strcmp(IoUmode, 'union') || strcmp(IoUmode, 'subject')), 'IoUmode for DenseCap baseline should be either union or subject');

    % Path to pre-computed Lu16 scores
    path_scores = sprintf('%s/%s/%s/%s', opts.dataroot, '%s', 'test', 'densecap-candidates');


    % UNREL
    opts.dataset = 'unrel-dataset';
    dataset = Dataset(opts);
    % Load candidates/annotations
    unrel.annotations = dataset.get_full_annotations();
    % Load pre-computed scores
    [unrel.pairs, unrel.scores] = load_densecap_outputs(sprintf(path_scores, opts.dataset));


    % VRD
    opts.dataset = 'vrd-dataset';
    dataset = Dataset(opts);
    % Load candidates/annotations
    vrd.annotations = dataset.get_full_annotations();
    % Load pre-computed scores
    [vrd.pairs, vrd.scores] = load_densecap_outputs(sprintf(path_scores, opts.dataset));


    % Merge dataset
    [candidates, scores, annotations] = merge_datasets_for_retrieval(unrel, vrd); 


    % Compute AP
    [ap, ub] = evaluate_retrieval_DenseCap(candidates, scores, annotations, opts);
    fprintf('mAP=%.1f\n', 100*mean(ap));

end

function [ap, ub] = evaluate_retrieval_DenseCap(candidates, scores, annotations, opts)

    overlap_thresh  = opts.overlap;
    IoUmode         = opts.IoUmode;
    load(fullfile(opts.dataroot, opts.vocab_objects));
    load(fullfile(opts.dataroot, opts.vocab_predicates));
    load(fullfile(opts.dataroot, opts.unusual_triplets));

    ap = zeros(length(triplets),1);
    ub = zeros(length(triplets),1); % provide upperbound detection. Number of hits.

    for t=1:length(triplets)

        % Extract the groundtruth positives, the candidates and scores for the triplet query	
        triplet                     = Dataset.get_triplet_index(triplets{t}, vocab_objects, vocab_predicates);
        positives_triplet           = Dataset.get_annotations_for_triplet(annotations, triplet); 
        scores_triplet              = scores(:, t); % the score of the triplet is the score of the predicate
        [ap(t), ub(t)]              = compute_ap(candidates, scores_triplet, positives_triplet, overlap_thresh, IoUmode);

    end
end


function [regions, triplet_scores] = load_densecap_outputs(path)


    regions = struct('im_id',[],'reg_id',[],'subject_box',[], 'object_box',[]);
    triplet_scores = [];

    files = dir(fullfile(path, 'candidates', '*.mat'));
    num_images = length(files);

    reg_id = 1;
    for i=1:num_images

       boxes = load(fullfile(path, 'candidates', files(i).name));
       region_box = boxes.x';

       num_candidates = size(region_box,1);

       im_id = strsplit(files(i).name, '.');
       im_id = str2num(im_id{1});

       regions.im_id = [regions.im_id ; im_id*ones(num_candidates,1)];
       regions.reg_id = [regions.reg_id ; reg_id*ones(num_candidates,1)];
       regions.subject_box = [regions.subject_box ; region_box];

       % Load triplet scores
       scores = load(fullfile(path, 'scores', files(i).name));
       scores = scores.x';

       triplet_scores = [triplet_scores ; scores];

       reg_id = reg_id + 1;

    end

    regions.object_box = regions.subject_box;

    % Minus score as DenseCap returns distance
    triplet_scores = -triplet_scores; 


end




