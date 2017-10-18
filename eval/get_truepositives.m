function [hit, pos] = get_truepositives(candidates, gtpositives, overlap_thresh, IoUmode)

	npos = length(gtpositives.im_id);
	ncandidates = length(candidates.im_id);
	hit = zeros(npos,1);
	pos = zeros(ncandidates,1);

	for j=1:ncandidates

		data_id = candidates.data_id(j);
		im_id   = candidates.im_id(j);
		sub_box = candidates.subject_box(j,:);
		obj_box = candidates.object_box(j,:);

		% Get the groundtruth positives in the same image
		idx = find(gtpositives.data_id==data_id & gtpositives.im_id==im_id);

		% Compute best overlap of the current proposal with the groundtruth positives (not already hit)
		ovlmax  = -inf;
		k_hit   = -1;

		for k=1:length(idx)

			if hit(idx(k)) ==1 % skip if already hit
				continue
			end

			sub_box_gt = gtpositives.subject_box(idx(k),:);
			obj_box_gt = gtpositives.object_box(idx(k),:);

			switch IoUmode

				case 'union'
				union_box = get_unionbox(sub_box, obj_box);
				union_box_gt = get_unionbox(sub_box_gt, obj_box_gt);
				ovl = get_overlap(union_box, union_box_gt);

				case 'subject'
				ovl = get_overlap(sub_box, sub_box_gt);

				case 'subject-object'
				ovl_sub = get_overlap(sub_box, sub_box_gt);
				ovl_obj = get_overlap(obj_box, obj_box_gt);
				ovl     = min([ovl_sub, ovl_obj]);

				otherwise
				error('IoU mode %s not implemented\n',IoUmode);
			end

			if ovl >= overlap_thresh && ovl > ovlmax
				ovlmax  = ovl;
				k_hit   = k;
			end

		end

		if k_hit > 0
			pos(j) = 1;
			hit(idx(k_hit)) = 1;
		end

	end

	pos(pos==0) = -1;


end