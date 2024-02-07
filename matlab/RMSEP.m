function [retour, nombre]=RMSEP(yexp,ycalc)

% RMSEP            a pour fonction de calculer la racine carrée de 
%                   l'écart quadratique moyen entre valeurs 
%                  expérimentales et valeurs calculées. Typiquement ce 
%                  critère permet de quantifier la qualité d'un 
%                  ajustement
% OUTPUT
% retour           ='Root Mean Square Error' 
% 
% INPUT
% ycalc            = propriete calculee (vecteur) 
% yexp             = propriete experimentale (vecteur)

EP=yexp-ycalc;

nombre=max(size(EP));

if size(EP,1)==1
	retour=sqrt(EP*EP'/nombre);
else
	retour=sqrt(EP'*EP/nombre);
end
