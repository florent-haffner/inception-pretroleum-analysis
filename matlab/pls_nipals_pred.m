%# Function ypred=pls_nipals(X,Y,A,preproc,Xtest);
%#
%# AIM:         performs PLS calibration on X and Y
%# PRINCIPLE:   Uses the NIPALS algorithm to perform PLS model calibration
%# REFERENCE:   Multivariate Calibration, H. Martens, T. Naes, Wiley and
%#              sons, 1989
%#
%# INPUT:
%# X            matrix of independent variables (e.g. spectra) for the 
%#              calibration set (n x p)
%# Y            vector of y reference values (n x 1) for the calibration
%#              set
%# A            number of PLS factors to consider
%# preproc      preprocessing applied to data
%#              0: no preprocessing
%#              1: column mean-centering of X and Y
%# Xtest        matrix of independent variables for the test set (ntest x p)
%#
%# OUTPUT:
%# ypred        y for the test set predicted by the model with A factors
%#
%# AUTHOR:      Xavier Capron
%# 			    Copyright(c) 2004 for ChemoAC
%# 			    FABI, Vrije Universiteit Brussel
%# 			    Laarbeeklaan 103, 1090 Jette
%# 			    Belgium
%#             
%# VERSION: 1.0 (24/11/2004)

function ypred=pls_nipals_pred(X,Y,A,preproc,Xtest)

n=size(X,1);

if preproc==1        
        [X,mX]=center(X,1);
        [Y,mY]=center(Y,1);
end

Xorig=X;
Yorig=Y;

for a=1:A
    
    c=(Y'*X*X'*Y)^(-0.5);
    W(:,a)=c*X'*Y;
    T(:,a)=X*W(:,a);
    P(:,a)=X'*T(:,a)/(T(:,a)'*T(:,a));
    Q(a,1)=Y'*T(:,a)/(T(:,a)'*T(:,a));
    X=X-T(:,a)*P(:,a)';
    Y=Y-T(:,a)*Q(a,1);
    
end

B=W*(P'*W)^(-1)*Q;

n_test=size(Xtest,1);

if preproc==1
    Xtest=Xtest-ones(n_test,1)*mX;
    ypred=Xtest*B+mY;
else
    ypred=Xtest*B;
end

