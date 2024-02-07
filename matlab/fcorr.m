function [Xtot,wtot]=FCORR(Xtot,wtot,optclb,lg,optco,lg1,lg2)

% [Xtot,wtot,Stot]=FCORR(Xtot,wtot,optclb,lg,optco,lg1,lg2);
% ENTREES
% Xtot = matrice des spectres en entrée
% wtot = longueurs d'onde correspondantes
% optclb = option correction ligne base (1=oui)
% lg = bande de longueurs d'onde sur laquelle on fait le traitement
%       de correction de ligne de base 
% optco  = option correction chemin optique (1=oui)
% lg1 = [lg11 lg12] bande de lo sur laquelle on cherche le 1er min
% lg2 = [lg21 lg22] bande de lo sur laquelle on cherche le 2nd min 
% SORTIES
% Xtot = matrice des spectres corrigés (et/ou) normalisés
% wtot = longueurs d'onde correspondantes
%
% Algorithme : Weighted Least Squares Baseline
%
% La correction de ligne de base (CLB) s'effectue habituellement sur  
% la gamme lg de longueurs d'onde, sous-partie de wtot 
% Les spectres renvoyés sont de longueur lg.
% La CLB consiste à trouver la droite définie sur lg qui fitte le mieux le
% spectre et à la sosutraire du spectre. Le spectre corrigé obtenu a pour
% ligne de base 0, et des absorbances corrigées positives et négatives
%
% La correction de chemin optique consiste à normaliser les spectres
% donc à ramener la surface totale du spectre (son énergie) à 1. la surface du spectre 
% est la surface comprise sur un intervalle de lo
% entre le spectre et la ligne du 0 : cette ligne du 0 n'est pas la ligne
% de base, mais la droite en-dessous de laquelle le spectre ne descend pas
% : tout le spectre est au-dessus de cette ligne du 0. Pour la définir, on
% recherche les points de tangence du spectre avec cette ligne. Ces points
% de tangence se situent pour le 1er sur l'intervalle lg1 et pour le second
% sur l'intervalle lg2

%*********************************************
% on selectionne les longueurs d'onde
%*********************************************

% wtot en vecteur ligne
if size(wtot,1) > 1
    wtot=wtot';
end

sel=[];
for i=1:size(lg,1)
	seld=max(find(wtot <= lg(i,1) ));
	self=min(find(wtot >= lg(i,2) ));
    if isempty(seld)
        seld=max(find(wtot == min(wtot) ));
    end
    if isempty(self)
        self=min(find(wtot == max(wtot) ));
    end
	sel=[sel seld:self];
end

Xtot=Xtot(:,sel);
wtot=wtot(sel);

clear seld self sel

%*********************************************
% choix de la bande spectrale
% absorbance la + petite et la + grande
%*********************************************
if nargin ==7
	if optco==1
		Xtot=Xtot';

		lg11=min(lg1); lg12=max(lg1);
		ia=find(wtot> lg11 & wtot< lg12);
		a1=min(Xtot(ia,:));
		for i=1:max(size(a1))
			% indice dans ia
			ia1=min(find( Xtot(ia,i) == a1(i) ) );
			w1(i)=wtot(ia(ia1));
		end

		lg21=min(lg2); lg22=max(lg2);
		ia=find(wtot> lg21 & wtot< lg22);
		a2=min(Xtot(ia,:));
		for i=1:max(size(a2))
			% indice dans ia
			ia2=min(find( Xtot(ia,i) == a2(i) ) );
			w2(i)=wtot(ia(ia2));
		end

		Xtot=Xtot';

%*********************************************
% placement de la ligne de base
%*********************************************
		for i=1:size(Xtot,1)
			pente=(a2(i)-a1(i) )/(w2(i)-w1(i));
			Xtot(i,:)=Xtot(i,:)- (pente*(wtot-w1(i))+a1(i));
		end

		clear ia
		clear ia1 ia2 a1 a2 w1 w2
		clear pente i

%*********************************************
% on normalise pour le chemin optique
% sur la selection de longueurs d'onde sel
%*********************************************

		%Stot=sum(Xtot(:,sel)');
		Stot=sum(Xtot');
		Stot=Stot';
 
        Xtot=Xtot./(Stot*ones(1,size(Xtot,2)));

		clear Stot
	end
end

%*********************************************
% correction de ligne de base
%*********************************************
if optclb==1
	wtotc=wtot'-mean(wtot') ;
	wtotn=wtotc/sqrt(wtotc'*wtotc);
	for i=1:size(Xtot,1)
	  Xm=mean(Xtot(i,:));
      Xc=(Xtot(i,:)-Xm)';
      Xtot(i,:)= ( Xc- wtotn*(wtotn'*Xc) )';
	end

	clear wtotc wtotn
	clear Xc Xm
end
