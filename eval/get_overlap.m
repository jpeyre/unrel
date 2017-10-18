function ovl = get_overlap( box1, box2 )

% Compute the overlap (intersection over union) between 2 boxes
% The boxes should be given as [xmin, ymin, xmax, ymax]

% Intersection
boxI = [max(box1(1),box2(1)),...
	max(box1(2),box2(2)),...
	min(box1(3),box2(3)),...
	min(box1(4),box2(4)) ];

wI = boxI(3) - boxI(1) + 1;
hI = boxI(4) - boxI(2) + 1;

ovl = 0;

if wI>0 && hI>0 
    
    % Compute area of the union
    area_union = (box1(3)-box1(1)+1)*(box1(4)-box1(2)+1) + (box2(3)-box2(1)+1)*(box2(4)-box2(2)+1) - wI*hI;
    
    % Compute IoU
    ovl = wI*hI / area_union;
    
end

end

