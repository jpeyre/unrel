function A = pairs_in_bag(bags, pairs)


num_pairs = length(pairs.im_id);
A 	  = sparse(length(bags), num_pairs);

for i = 1:length(bags)
	im_id 	  = bags(i).im_id;
	sub_cat	  = bags(i).sub_cat;
	obj_cat	  = bags(i).obj_cat;
	indim     = find(pairs.im_id==im_id); % get the pairs for this image
	idx	  = pairs.sub_cat(indim)==sub_cat & pairs.obj_cat(indim)==obj_cat;
	idx 	  = indim(idx);
	A(i, idx) = 1;
end



end
