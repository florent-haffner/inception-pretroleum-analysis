function Xscale=rscalex(X,Xsmin,Xsmax);

[n,p]=size(X);

Xmin=ones(n,1)*min(X);
rX=ones(n,1)*range(X);
rXs=ones(n,p)*(Xsmax-Xsmin);

Xscale=(X-Xmin).*(rXs./rX)+Xsmin;
