function [features] = extract_spatialconfig(pairs)

  [x1,y1,w1,h1,x2,y2,w2,h2] = get_boxes_xywh(pairs);

  % Scale
  scale1 = w1.*h1;
  scale2 = w2.*h2;

  % Offset
  x1_c 	= x1+w1/2;
  y1_c	= y1+h1/2;
  x2_c	= x2+w2/2;
  y2_c	= y2+h2/2;
  offsetx = x2_c-x1_c;
  offsety = -(y2_c-y1_c);

  % Aspect ratio
  aspectx = w1./h1;
  aspecty = w2./h2;

  % Overlap
  boxI.xmin = max(x1,x2);
  boxI.ymin = max(y1,y2);
  boxI.xmax = min(x1+w1-1, x2+w2-1);
  boxI.ymax = min(y1+h1-1, y2+h2-1);
  wI 	    = max((boxI.xmax - boxI.xmin + 1), 0);
  yI 	    = max((boxI.ymax - boxI.ymin + 1), 0);
  areaI     = wI.*yI;
  areaU	    = scale1 + scale2 - areaI;


  % Fill spatial features
  N = length(x1);
  features 	= zeros(N,6);
  features(:,1) = offsetx./sqrt(scale1);
  features(:,2) = offsety./sqrt(scale1);
  features(:,3)	= sqrt(scale2./scale1);
  features(:,4)	= aspectx;
  features(:,5)	= aspecty;
  features(:,6)	= sqrt(areaI./areaU);

end


function [x1,y1,w1,h1,x2,y2,w2,h2] = get_boxes_xywh(pairs)

  x1 = pairs.subject_box(:,1);
  y1 = pairs.subject_box(:,2);
  w1 = pairs.subject_box(:,3) - pairs.subject_box(:,1) + 1;
  h1 = pairs.subject_box(:,4) - pairs.subject_box(:,2) + 1;
  x2 = pairs.object_box(:,1);
  y2 = pairs.object_box(:,2);
  w2 = pairs.object_box(:,3) - pairs.object_box(:,1) + 1;
  h2 = pairs.object_box(:,4) - pairs.object_box(:,2) + 1;

end






