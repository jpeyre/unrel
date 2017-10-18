function vis_retrieval()

	% Visualize the top retrieved candidate pairs given a triplet query
	% For this, put the images of each dataset in subfloder : ./data/datasetname/images
	opts = config();
	opts.split = 'test';
	opts.use_languagescores = 0;
	opts.use_objectscores = 0;
	opts.IoUmode = 'subject-object'; % choose 'union', 'subject, 'subject-object'
	opts.candidatespairs = 'candidates'; opts.overlap = 0.3;
	opts.supervision = 'weak';

	% Load a pre-trained model, e.g : 
	load(fullfile(opts.dataroot, 'classifiers', [opts.supervision, '_', strjoin(opts.featurestype, '-'), '_bg', num2str(opts.num_negatives)]));

	datasets = {'unrel-dataset','vrd-dataset'};
	image_filenames = [];
	for k=1:length(datasets)
		path = load(fullfile(opts.dataroot, datasets{k},['image_filenames_', opts.split, '.mat']));
		image_filenames{k} = path.image_filenames;
	end


	load(fullfile(opts.dataroot, opts.vocab_objects));
	load(fullfile(opts.dataroot, opts.vocab_predicates));
	load(fullfile(opts.dataroot, opts.unusual_triplets));

	%% Compute scores

	opts.dataset = datasets{1};
	[unrel.pairs, unrel.scores, unrel.annotations] = predict(W, opts);
	opts.dataset = datasets{2};
	[vrd.pairs, vrd.scores, vrd.annotations] = predict(W, opts);

	% Merge dataset
	[pairs, scores, annotations] = merge_datasets_for_retrieval(unrel, vrd); 

	%% Get positives for triplet 

	% Choose triplet to visualize 'subject-predicate-object'
	tripletname = 'cat-wear-tie';
	%tripletname = triplets{1};

	% Get positives
	triplet = Dataset.get_triplet_index(tripletname, vocab_objects, vocab_predicates);
	positives_triplet = Dataset.get_annotations_for_triplet(annotations, triplet); 
	[candidates_triplet, idx]   = Dataset.get_candidates_for_triplet(pairs, triplet); 
	scores_triplet    = scores(idx, triplet.rel);
	[~, indsort] = sort(scores_triplet, 'descend');
	candidates_triplet = Dataset.select(candidates_triplet, indsort);
	[positives, labels] = get_truepositives(candidates_triplet, positives_triplet, opts.overlap, opts.IoUmode);
	

	%% Top results

	idx = find(labels==1); % top retrieved positives
	%idx = find(labels==-1); % top retrieved negatives

	figure(1)
	for j=1:min(length(idx),5)

		clf
		k = idx(j); 
		im_id       = candidates_triplet.im_id(k);
		data_id     = candidates_triplet.data_id(k);
		subject_box = candidates_triplet.subject_box(k,:);
		object_box  = candidates_triplet.object_box(k,:);
		filename	= image_filenames{data_id+1}{im_id};
		img			= imread(fullfile(opts.dataroot, datasets{data_id+1}, 'images', filename));
		% Resize the image
		scale = 500/size(img,1);
		img = imresize(img, scale);
		imshow(img);hold on,
		Boxes = [subject_box; object_box];
		Boxes = Boxes*scale;
		draw_boxes(Boxes);
		pause

	end


	%% Missed detections

	idx = find(positives==0); % missed detections

	figure(1)
	for j=1:min(length(idx),5)

		clf
		k = idx(j); 
		im_id       = positives_triplet.im_id(k);
		data_id     = positives_triplet.data_id(k);
		subject_box = positives_triplet.subject_box(k,:);
		object_box  = positives_triplet.object_box(k,:);
		filename	= image_filenames{data_id+1}{im_id};
		img			= imread(fullfile(opts.dataroot, datasets{data_id+1}, 'images', filename));
		% Resize the image
		scale = 500/size(img,1);
		img = imresize(img, scale);
		imshow(img);hold on,
		Boxes = [subject_box; object_box];
		Boxes = Boxes*scale;
		draw_boxes(Boxes);
		pause

	end

end

function draw_boxes( Boxes, linewidth)

	% Display boxes on current figure
	% Boxes is a matrix : B * 4 where B is the number of boxes to draw
	% The coordinates have to been given in the format : [xmin, ymin, xmax, ymax]

	if nargin<2
		linewidth=8;
	end

	B = size(Boxes,1);

	if B==2
		colormap = [[0,0,1];[1,0,0]];
	elseif B==1
		colormap = [[1,0,0]];
	else	
		colormap = jet(B);
	end

	% Convert coordinates [x,y,w,h]
	Boxes(:,3) = Boxes(:,3) - Boxes(:,1) + 1;
	Boxes(:,4) = Boxes(:,4) - Boxes(:,2) + 1;

	for b=1:B
		box = Boxes(b,:); 
		rectangle('Position',box, 'EdgeColor',colormap(b,:),'LineWidth',linewidth); hold on,
	end

	hold off

end

