function [bags, image_bags, image_blocks] = form_blocks_pairs(pairs, annotations)

	% Inputs : candidate pairs of boxes
	%          annotations
	% Outputs : bags : 1 bag = 1 image-level annotation (e.g. : there is "person ride horse" in the first image)
	%           image_blocks : 1 block = 1 image. image_blocks{i} stores the indices of all candidate pairs for image i
	%           image_bags : image_bags{i} stores all the bags associated to image i

	
	%% Build the bags : 1 bag = 1 image-level annotation

	bags = struct();
	for i=1:length(annotations.im_id) 

		bags(i).im_id   = annotations.im_id(i);
		bags(i).sub_cat = annotations.sub_cat(i);
		bags(i).rel_cat = annotations.rel_cat(i);
		bags(i).obj_cat = annotations.obj_cat(i);

	end    

	%fprintf('%d unique bags found\n',length(bags));    

	%% Build the image blocks : image_blocks{i} return the indices of all pairs in image index i

	Images = [bags(:).im_id];
	Images = unique(Images); % images with at least one annotation 

	image_blocks = cell(length(Images),1); 
	count = 0;
	for i=1:length(Images)

	   im_id = Images(i);
	   idx = find(pairs.im_id==im_id)';
	   if length(idx)>0
		count = count + 1;
		image_blocks{count} = idx;
	   end
	end
	image_blocks = image_blocks(1:count); 

	%fprintf('%d image blocks constructed\n',length(image_blocks));


	%% Build image bags : image_bags{i} return all the bags (annotations) for image index i

	image_bags = cell(length(Images),1); 
	for b = 1:length(bags)

		index = find(Images == bags(b).im_id);
		image_bags{index} = [image_bags{index}, b];

	end

end
