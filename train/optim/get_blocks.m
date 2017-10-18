function result = get_blocks(image_blocks, n, P)

	result = [];

	for i=0:P-1
    		result = cat(2,result,image_blocks+n*i);
	end


end
