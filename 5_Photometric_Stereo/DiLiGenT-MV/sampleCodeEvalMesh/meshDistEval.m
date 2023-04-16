function [Dists,ID] = meshDistEval(vertices, vertices_Gt)
%% Evaluate how well a reconstructed 3D surface is.
%  Input: veritices_Gt and vertices should be (numV*3) vectors, where numV is
%  number of vertices;
%  Output: Dists(numV*1) and ID(numV*50). 

[ID, Dists] = knnsearch(vertices_Gt, vertices, 'K', 50);
Dists = mean(Dists, 2);

end

