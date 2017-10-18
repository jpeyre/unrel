function [pairs, scores, annotations] = merge_datasets_for_retrieval(dataset1, dataset2)

    % Merge the datasets for retrieval. 
    %To keep track of the dataset we append a field dataid to candidates and annotations
    dataset1.pairs.data_id       = zeros(Dataset.num(dataset1.pairs),1);
    dataset2.pairs.data_id       = ones(Dataset.num(dataset2.pairs),1);
    dataset1.annotations.data_id = zeros(Dataset.num(dataset1.annotations),1);
    dataset2.annotations.data_id = ones(Dataset.num(dataset2.annotations),1);

    pairs       = Dataset.merge_struct([dataset1.pairs, dataset2.pairs]);
    scores      = [dataset1.scores; dataset2.scores];
    annotations = Dataset.merge_struct([dataset1.annotations, dataset2.annotations]);

end
