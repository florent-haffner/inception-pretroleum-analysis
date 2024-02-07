%# Function mat_dist=distx(X);
%#
%# AIM: compute all possible euclidean distance between the samples (rows) of matrix X
%# PRINCIPLE: Uses the Al-Kashi's theorem to compute the euclidean distances
%# INPUT:
%# X            matrix of data (n x p)
%# 
%# OUTPUT:
%# mat_dist     matrix of distances (n x n)
%# 
%# AUTHOR:  Xavier Capron
%# 			Copyright(c) 2004 for ChemoAC
%# 			FABI, Vrije Universiteit Brussel
%# 			Laarbeeklaan 103, 1090 Jette
%# 			Belgium
%#             
%#          VERSION: 1.0 (24/11/2004)

function mat_dist=distx(X);

[n,p]=size(X);

mat_dist=zeros(n,n);

XX=X*X';

for i=1:n
    for j=i+1:n
        mat_dist(i,j)=XX(i,i)+XX(j,j)-2*XX(i,j);
    end
end

mat_dist=sqrt(mat_dist);
mat_dist=mat_dist+mat_dist';

return
