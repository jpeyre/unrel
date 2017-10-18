function opts = config(varargin)

    ip = inputParser;
    ip.addOptional('dataroot', './data', @isstr); % folder where data are located
    ip.addOptional('featurestype', {'spatial','appearance'}, @isstruct); % visual features to use : choose {'spatial'}, {'appearance'}, {'spatial', 'appearance'}
    ip.addOptional('dataset', 'vrd-dataset', @isstr); % dataset on which to train/evaluate e.g. 'vrd-dataset', 'unrel-dataset'
    ip.addOptional('annotatedpairs', 'annotated' , @isstr); % folder to annotated pairs. E.g. "annotated"
    ip.addOptional('candidatespairs', 'candidates' , @isstr); % folder to candidates pairs. E.g. "candidates", "Lu-candidates" 
    ip.addOptional('vocab_objects', fullfile('vrd-dataset','vocab_objects.mat'), @ischar);
    ip.addOptional('vocab_predicates', fullfile('vrd-dataset','vocab_predicates.mat'), @ischar);
    ip.addOptional('unusual_triplets', fullfile('unrel-dataset','annotated_triplets.mat'), @ischar);    
    ip.addOptional('language_model', fullfile('classifiers','Lu-language.mat'), @ischar); % language model trained by Lu16
    ip.addOptional('word2vec', fullfile('vrd-dataset','obj2vec.mat'), @ischar); % path to word2vec features used by Lu16

    % Train options
    ip.addOptional('supervision', 'full', @isstr); % 'weak' (using diffrac) or 'full' (ridge regression)
    ip.addOptional('num_negatives', 0 , @isscalar); % 0 if training without background, else fraction of negatives to be added
    
    % Testing options
    ip.addOptional('zeroshot', 0 , @isscaler); % 1 if zeroshot
    ip.addOptional('use_visualscores', 1 , @isscaler); % put 1 to use the visual scores 
    ip.addOptional('use_languagescores', 0 , @isscaler); % put 1 to combine with language scores of [31]
    ip.addOptional('use_objectscores', 1 , @isscaler); % put 1 to combine with object scores
    ip.addOptional('Nre', 50 , @isscaler); % k in recall@k
    ip.addOptional('overlap', 0.3 , @isscaler); % IoU threshold for evaluation
    ip.addOptional('IoUmode', 'union', @isscaler); % IoU mode for evaluation on retrieval UnRel (choose 'union', 'subject', 'subject-object')    

    % Parameters
    ip.addParameter('num_predicates', 70, @isscalar); % R in the paper
    ip.addParameter('n_iter', 1e5, @isscalar); % number of iterations
    ip.addParameter('lambda', 1e-6, @isscalar); % regularization parameter (Eq 3)
    ip.addParameter('bias', 100, @isscalar); % to avoid regularizing the bias
    ip.addParameter('alpha', 1.5, @isscalar); % lowerbound constraint (Eq 4), taking alpha>=1 empirically works better
    ip.addParameter('kapa', 1e-4, @isstr); % slack penalization
    ip.addParameter('alpha_predicates', 1, @isscalar); % predicate score weight (Eq 5)
    ip.addParameter('alpha_objects', 0.5, @isscalar); % object score weight (Eq 5)
    ip.addParameter('alpha_language', 0.07, @isscalar); % language score weight (Eq 5)
  
    ip.parse(varargin{:});
    opts = ip.Results;

end
	
