%# function list=kenstonx(X,n_object,opt);
%# 
%# AIM:         Representative subset selection of samples
%# PRINCIPLE:   Uses the Kennard and Stone algorithm to select the most representative samples of a data set
%# REFERENCE:   R. W. Kennard and L. A. Stone, Technometrics Vol. 11, No. 1, 1969 		
%# INPUT: 
%# X            matrix of data (n x p)
%# n_object     number of samples to be selected by the algorithm (between 1 and n)
%# opt          position of the first selected sample (1: closest to the mean, 0: farthest from the mean)
%# 
%# OUTPUT:
%# list         list of selected samples
%# 
%# SUBROUTINES: distx.m
%#
%# AUTHOR:  Xavier Capron
%# 			Copyright(c) 2004 for ChemoAC
%# 			FABI, Vrije Universiteit Brussel
%# 			Laarbeeklaan 103, 1090 Jette
%# 			Belgium
%#             
%#          VERSION: 2.0 (24/11/2004)


function list=kenstonx(X,n_object,opt);

[n,p]=size(X);

if nargin<3
    opt=1;
end

if n_object==0
    list=[];
    return
end

if n_object>n
    disp('Number of selected objects cannot be bigger than total number of objects !');
    disp(['Number of selected objects = ' num2str(n)])
    n_object=n;
end

index=[1:n]';

%trouve l'object le plus proche/le plus éloigné de la moyenne (1er objet);
mX=mean(X);
mX_mat=ones(n,1)*mX;

mX_dist=sqrt(sum((X-mX_mat).^2,2));
if opt==1
    [min_dist,first]=min(mX_dist);
else
    [max_dist,first]=max(mX_dist);
end

list=first;
if n_object==1
    return
else
    %calcul des distances entre tous les points
    dist_mat=distx(X);
    
    %trouve le 2e objet (le plus eloigné du 1er)
    [max_dist,second]=max(dist_mat(:,list));
    
    list=[list;second];
    index(list)=[];
 
    for i=3:n_object
        [min_dist,ind_min]=min(dist_mat(list,index),[],1);
        [max_dist,ind_max]=max(min_dist);
        obj=ind_max;
        list=[list;index(obj)];
        index(obj)=[];
    end
    
end



