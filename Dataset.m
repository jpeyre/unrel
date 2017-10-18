classdef Dataset

    % Class containing methods to load annotations, candidate pairs and features

    properties

        dataroot            = [];
        name                = []; 
        split               = []; 
        datapath            = [];
        candidatespairs     = [];
        annotatedpairs      = [];
        supervision         = [];
        vocab_objects       = [];
        vocab_predicates    = [];
	word2vec            = [];

    end

    methods

        function data = Dataset(opts)

            data.dataroot            = opts.dataroot;
            data.name                = opts.dataset;
            data.split               = opts.split;	
            data.datapath            = fullfile(data.dataroot, data.name, data.split);
            data.candidatespairs     = opts.candidatespairs;
            data.annotatedpairs      = opts.annotatedpairs;
            data.supervision         = opts.supervision;
            load(fullfile(data.dataroot, opts.vocab_objects));
            data.vocab_objects       = vocab_objects;
            load(fullfile(data.dataroot, opts.vocab_predicates));
            data.vocab_predicates    = vocab_predicates;         
            data.word2vec            = opts.word2vec;
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%% Loading candidates %%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		function pairs = load_candidates(data, candidatestype)
			% Load candidate pairs of boxes
			load(fullfile(data.datapath, candidatestype, 'pairs.mat'));
		end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%% Loading annotations %%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		function annotations = get_full_annotations(data)
			% Load box-level annotations
			annotations = data.load_candidates(data.annotatedpairs);
		end


		function annotations = get_weak_annotations(data)
			% Transform box-level annotations into image-level annotations
			full_annotations = data.load_candidates(data.annotatedpairs);
	
			triplets = [full_annotations.im_id, full_annotations.sub_cat, full_annotations.rel_cat, full_annotations.obj_cat];
			triplets = unique(triplets, 'rows');

			annotations = struct('im_id',   triplets(:,1),...
							 'sub_cat', triplets(:,2),...
							 'rel_cat', triplets(:,3),...
							 'obj_cat', triplets(:,4) );

		end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%% Load and process candidates for training %%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		function [X, Y, pairs] = load_traindata_full(data, positivepairs, featurestype, num_negatives, negativepairs)

			% Load data to train with full supervision
			% Input : positivepairs : folder name containing the positive pairs
			%         negativepairs  : folder name containing candidates from which to sample negatives
			%         featurestype   : type of features to load
			%         num_negatives  : 0 if train without background, else number of negative pairs to sample
			% Output : X     : [Nxd] matrix of visual features
			%          Y     : [NxR] binary matrix of labels for the predicates
			%          pairs : structure corresponding to the selected pairs of boxes


			% Load the annotated pairs
			pairs = data.load_candidates(positivepairs);

			% Process the pairs (remove duplicates + put annotations in matrix format)
			num_classes = length(data.vocab_predicates);
			[Y, pairs]  = data.binarize_label(pairs, num_classes);
			X           = data.load_visualfeatures(positivepairs, pairs, featurestype);

			% Sample the negatives (if background)
			if num_negatives > 0

				fprintf('Sampling negatives for training with background class...\n');

				% Sample negatives from candidates pairs
				annotations    = data.get_weak_annotations();
				pairs_neg      = data.load_candidates(negativepairs);
				[pairs_neg, ~] = data.sample_negative_candidates(pairs_neg, annotations, num_negatives);
				X_neg          = data.load_visualfeatures(negativepairs, pairs_neg, featurestype);

				% Merge with positives
				X                 = [X ; X_neg];
				pairs_neg.rel_cat = zeros(data.num(pairs_neg),1); % add background label to the negative pairs
				pairs             = data.merge_struct([pairs ; pairs_neg]); 
				pairs.rel_cat     = pairs.rel_cat + 1; % Increment number of classes ("no relation" class)
				num_classes       = num_classes + 1;

				% Remove duplicates and binarize label
				[Y, pairs, idTo_pairs] = data.binarize_label(pairs, num_classes);
				X = X(idTo_pairs, :);

			end
             
        end


		function [X, annotations, pairs] = load_traindata_weak(data, positivepairs, featurestype, num_negatives, negativepairs)

			% Load data to train with weak supervision
			% Input : positivepairs : folder name containing the positive pairs
			%         negativepairs  : folder name containing candidates from which to sample negatives
			%         featurestype   : type of features to load
			%         num_negatives  : 0 if train without background, else number of negative pairs to sample
			% Output : X           : [Nxd] matrix of visual features
			%          annotations : structure storing the image-level annotations
			%          pairs       : structure corresponding to the selected pairs of boxes


			% Load image-level annotations
			annotations = data.get_weak_annotations();

			% Load the positive candidates (filtering with image-level labels)
			pairs = data.load_candidates(positivepairs);
			[pairs, annotations] = data.get_positive_candidates(pairs, annotations);

			% Get the visual features
			X = data.load_visualfeatures(positivepairs, pairs, featurestype);	

			% Sample the negatives (if background)
			if num_negatives > 0

				fprintf('Sampling negatives for training with background class...\n');
				pairs_neg = data.load_candidates(negativepairs);
				[pairs_neg, annotations_neg] = data.sample_negative_candidates(pairs_neg, annotations, num_negatives);
				X_neg = data.load_visualfeatures(negativepairs, pairs_neg, featurestype);

				% Merge with posivites
				X           = [X ; X_neg];
				pairs       = data.merge_struct([pairs ; pairs_neg]);
				annotations = data.merge_struct([annotations ; annotations_neg]);
				annotations.rel_cat = annotations.rel_cat + 1;

			end
        end



 	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%b
	%%%%%%%% Pairs filtering methods %%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      


		function [pairs, annotations] = get_positive_candidates(data, pairs, annotations)

			% Input : pairs       : all candidate pairs of boxes
			%         annotations : all image-level annotations
			% Output : pairs       : candidates matching image-level labels
			%          annotations : image-level annotations which have at least 1 candidate pairs


			B = data.num(annotations); 
			ind_candidates = [];
			ind_pairs = [];

			for j=1:B
				im_id   = annotations.im_id(j);
				sub_cat = annotations.sub_cat(j);
				obj_cat = annotations.obj_cat(j);

				idx = data.get_pairs_in_image(pairs, im_id);
				indmatch = (pairs.sub_cat(idx) == sub_cat) & (pairs.obj_cat(idx) == obj_cat);
				indmatch = idx(indmatch);

				if length(indmatch) >= 1
					ind_candidates = [ind_candidates ; j];
					ind_pairs      = [ind_pairs ; indmatch];
				end
			end

			pairs = data.select(pairs, unique(ind_pairs));
			annotations = data.select(annotations, ind_candidates);
		end


		function [pairs, annotations_neg] = sample_negative_candidates(data, pairs, annotations, num_negatives)

			% Input : pairs         : candidates from which to sample the negatives
			% 	      annotations   : image-level annotations
			%         num_negatives : at most num_negatives is sampled
			% Output : pairs           : negative candidates (that do not match the image-level annotations)
			%          annotations_neg : structure containing negative annotations

			rng(1); % random seed to reproduce result

			% Get the negative candidates
			[pairs_pos, ~] = data.get_positive_candidates(pairs, annotations);
			[~, indneg] = setdiff(pairs.rel_id, pairs_pos.rel_id);

			% Sample at most num_negatives of them
			perm        = indneg(randperm(length(indneg)));
			indneg      = perm(1:min(length(indneg), num_negatives));
			pairs       = data.select(pairs, indneg);

			% Build annotation for negatives (mapped to background class)
			annotations_neg = struct('im_id',   pairs.im_id,...
										 'sub_cat', pairs.sub_cat,...
										 'rel_cat', zeros(data.num(pairs),1),...
										 'obj_cat', pairs.obj_cat);

		end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% Load features %%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		function [features] = load_visualfeatures(data, candidatestype, pairs, featurestype)

			datafolder = sprintf('%s/%s/%s/%s', data.datapath, candidatestype, 'features', ['%s', '-', data.supervision]);
			features = [];
			for k=1:length(featurestype)
				switch featurestype{k}

					case 'spatial'
						%fprintf('Loading spatial features...\n');
						pathfeats = sprintf(datafolder, 'spatial');
						feats = data.load_spatialfeatures(pathfeats, pairs);

					case 'appearance'
						 %fprintf('Loading appearance features...\n');
						pathfeats = sprintf(datafolder, 'appearance');
						feats = data.load_appearancefeatures(pathfeats, pairs);	

					otherwise
						error('This feature type is not known. Define loading function.')
				end
				features = [features, feats];
			end

		end


		function [X] = load_spatialfeatures(data, pathfeatures, pairs)

			N = data.num(pairs);
			images = data.get_images(pairs);
			X = zeros(N, 400);

			for j=1:length(images)
				im_id      = images(j);
				indpairsim = data.get_pairs_in_image(pairs, im_id);
				load(fullfile(pathfeatures, [num2str(im_id), '.mat']));

				for p=1:length(indpairsim)
					rel_id = pairs.rel_id(indpairsim(p));
					idx    = find(spatial(:,1) == rel_id);
					if length(idx)==0
						error('Spatial feature %d not found', rel_id)
					end
					X(indpairsim(p),:) = spatial(idx, 2:end);
				end
			end
		end


		function [X] = load_appearancefeatures(data, pathfeatures, pairs)

			N = data.num(pairs);
			images = data.get_images(pairs);
			X = zeros(N, 300*2);

			for j=1:length(images)
				im_id	   = images(j);
				indpairsim = data.get_pairs_in_image(pairs, im_id);
				load(fullfile(pathfeatures, [num2str(im_id),'.mat']));

				for p=1:length(indpairsim)

					sub_id = pairs.sub_id(indpairsim(p));
					idx = find(appearance(:,1) == sub_id);
					if length(idx)==0
						error('Object feature %d not found', sub_id)
					end
					sub_feat = appearance(idx, 2:end);

					obj_id   = pairs.obj_id(indpairsim(p));
					idx	 = find(appearance(:,1) == obj_id);
					if length(idx)==0
						error('Object feature %d not found', obj_id)
					end
					obj_feat = appearance(idx, 2:end);

					X(indpairsim(p),:) = [sub_feat, obj_feat];
					X(indpairsim(p),:) = X(indpairsim(p),:) / norm(X(indpairsim(p),:)); % L2-normalize concatenation
				end
			end

		end


		function [X] = load_languagefeatures(data, pairs)
			% Load Word2Vec language representation for (subject, object) categories
			
			load(fullfile(data.dataroot, data.word2vec)); % Word2vec models to encode object categories

			N = data.num(pairs);
			X = zeros(N, 600);
			for j=1:N
				sub_cat = pairs.sub_cat(j);
				obj_cat = pairs.obj_cat(j);
				X(j,:)  = [obj2vec(data.vocab_objects{sub_cat}), obj2vec(data.vocab_objects{obj_cat})];
			end

			X = cat(2,X,ones(N,1)); % add bias column at the end

		end


		function [scores] = load_objectscores(data, datafolder, pairs)

			% Load object scores for the candidate pairs

			N = data.num(pairs);
			scores = zeros(N,1);
			objectscores = load(fullfile(datafolder, 'objectscores.mat'));
			objectscores = objectscores.scores;

			for j=1:N
				sub_id     = pairs.sub_id(j);
				sub_cat    = pairs.sub_cat(j);
				idx 	   = objectscores(:,1)==sub_id;
				sub_scores = objectscores(idx,2:end);
				obj_id     = pairs.obj_id(j);
				obj_cat    = pairs.obj_cat(j);
				idx        = objectscores(:,1)==obj_id;
				obj_scores = objectscores(idx,2:end);
				scores(j)  = sub_scores(sub_cat+1) + obj_scores(obj_cat+1) ; % first index is background
			end
		end

		end


	methods (Static)

	%%%%%%%%%%%%%%%%%%%%%%%%%
	%%%%%%% Utils %%%%%%%%%%%
	%%%%%%%%%%%%%%%%%%%%%%%%%

		function nsamples = num(pairs)
			nsamples = length(pairs.im_id);
		end

		function images = get_images(pairs)
			images = unique(vertcat(pairs.im_id));		
		end	

		function pairs = select(pairs, idx)
			pairs = structfun(@(x) (x(idx,:)), pairs, 'UniformOutput', false);
		end

		function idx = get_pairs_in_image(pairs, im_id)
			idx = find(pairs.im_id == im_id);
		end

		function idx = get_matching_triplets(pairs, sub_cat, rel_cat, obj_cat)
			idx = (pairs.sub_cat == sub_cat) & (pairs.rel_cat == rel_cat) & (pairs.obj_cat == obj_cat);
		end

		function idx = get_matching_objects(pairs, sub_cat, obj_cat)
			idx = (pairs.sub_cat == sub_cat) & (pairs.obj_cat == obj_cat);
		end

		function triplet_cat = get_triplet_index(triplet, vocab_objects, vocab_predicates)
			
			% Parse the triplet
			triplet  = strsplit(triplet, '-');
			subject  = triplet{1};
			relation = triplet{2};
			object   = triplet{3};

			% Get the indices
			sub_cat = find(ismember(vocab_objects, subject));
			obj_cat = find(ismember(vocab_objects, object));
			rel_cat = find(ismember(vocab_predicates, relation));

			if (length(sub_cat)==0) || (length(obj_cat)==0) || (length(rel_cat)==0)
					error('Triplet not found in vocabulary');
			end
			triplet_cat.sub = sub_cat;
			triplet_cat.rel = rel_cat;
			triplet_cat.obj = obj_cat;
		end

		function [pairs] = get_annotations_for_triplet(pairs, triplet)

			% Return the positive pairs for a given triplet  
			idx = Dataset.get_matching_triplets(pairs, triplet.sub, triplet.rel, triplet.obj);
			pairs = Dataset.select(pairs, idx);

			% We remove duplicate annotations
			rows            = [pairs.im_id, pairs.sub_cat, pairs.rel_cat, pairs.obj_cat, pairs.subject_box, pairs.object_box];
			[~, id_unique]  = unique(rows, 'rows');
			pairs = Dataset.select(pairs, id_unique);

		end

		function [pairs, indices] = get_candidates_for_triplet(pairs, triplet)

			% Return the positive pairs for a given triplet
			idx = Dataset.get_matching_objects(pairs, triplet.sub, triplet.obj);
			pairs = Dataset.select(pairs, idx);

			% Keep the unique (subject, object) pairs of boxes
			rows            = [pairs.im_id, pairs.sub_cat, pairs.obj_cat, pairs.subject_box, pairs.object_box];
			[~, id_unique]  = unique(rows, 'rows');
			pairs           = Dataset.select(pairs, id_unique);

			% Return index : careful idx(id_unique)
			idx = find(idx == 1);
			indices = idx(id_unique);

		end


		function [pairs_unique, idTo_pairs, idTo_unique] = unique_pairs(pairs)
			% Unique pairs of boxes 
			pairs_of_boxes = [pairs.im_id, pairs.sub_cat, pairs.obj_cat, pairs.subject_box, pairs.object_box];

			[pairs_unique, idTo_pairs, idTo_unique] = unique(pairs_of_boxes,'rows');

		end

		function merge_structure = merge_struct(structures)
			% Merge structure of arrays with same fields
			% Input is a list of structures (of arrays) to be merged
			names = fieldnames(structures);
			data = cellfun(@(f) (vertcat(structures.(f))), names, 'UniformOutput', false);
			merge_structure = cell2struct(data,names);

		end


                function [Y, pairs, idTo_pairs] = binarize_label(pairs, num_classes)

                        % Put labels in binary matrix and remove redundant pairs

                        % Get the unique pairs of boxes (unique im_id, sub_cat, obj_cat, boxes)
                        [~, idTo_pairs, idTo_unique] = Dataset.unique_pairs(pairs);

                        N = length(idTo_pairs);
                        Y = zeros(N, num_classes);

                        for j=1:length(idTo_unique)
                                idx = idTo_unique(j);
                                rel_cat = pairs.rel_cat(j);
                                Y(idx, rel_cat) = 1;
                        end

                        pairs = Dataset.select(pairs, idTo_pairs);

                end



	end

end
	
