function union_boxes = get_unionbox( box1, box2 )

% Compute union of the boxes given in format : [xmin, ymin, xmax, ymax]

union_boxes = [	min(box1(:,1), box2(:,1)),...
		min(box1(:,2), box2(:,2)), ...
                max(box1(:,3), box2(:,3)), ...
		max(box1(:,4), box2(:,4)) ];


end

