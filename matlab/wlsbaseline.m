function [data,allb,a] = wlsbaseline(data,order,options)
%WLSBASELINE Weighted least squares baseline function.
%  Subtracts a baseline (or other signal) from a spectrum using
%  an iterative asymmetric least squares algorithm. Points with
%  residuals <0 are up-weighted at each iteration of the least
%  squares fitting. This results in a robust "non-negaitve" residual
%  fit when residuals of significant amplitude (e.g. signals on a
%  background) are present.
%
%  INPUTS:
%        data = MxN spectral data to be baselined.
%    baseline = 1xN reference spectrum (or KxN spectra) to use
%               for baseline,
%               or an integer scalar value (order) corresponding to the
%               order of polynomial baseline to use.
%
%  OPTIONAL INPUT:
%     options = an options structure with the following fields:
%        plots: [{'none'} | 'debug' | 'intermediate' | 'final'] governs plots,
%   weightmode: [ {1} | 2 ] flag indicating which weighting mode to use.
%                Mode 1 = Power method. Negative (<0) residuals are weighted up
%                by the power of 10.^(option.negw). All residuals are then
%                raised to the power of (option.power).
%                Mode 2 = T squared method. Negative (<0) residuals are weighted
%                up by the extent to which the surpass an estimate of the
%                noise limit and the approximate t-limit defined by
%                (option.tsqlim).
%      trbflag: [ 'bottom' | 'top' ] Baseline to top or bottom of data.
%         negw: {1} upweighting scale of negative values (10^negw) (used only
%                for weightmode = 1),
%        power: {2} exponential amplification of residuals (used only for
%                weightmode = 1),
%       tsqlim: [0.99] t-test confidence limit for significant negative
%                residuals which need to be up-weighted. (used only for
%                weightmode = 2),
%       nonneg: [{'no'}|'yes'] flag to force non-negative baseline weighting,
%                Most often used when "real" spectra are used for baslineing
%                and they should not be "flipped" by a negative weighting.
%                Using nonneg = 'yes', WLSBASELINE an be used as a partial
%                CLS prediction to estimate the concentration of a species
%                when not all species' pure component spectra are known.
%        delta: [1e-9] change-of-fit convergence criterion,
%      maxiter: [100] maximum iterations allowed per spectrum
%      maxtime: [600] maximum time (in seconds) permitted for baselining of
%                all data.
%
%  OUTPUTS:
%      bldata = MxN baselined data.
%         wts = Weights corresponding the amount of baseline removed from
%               each spectrum (i.e. bldata = data - wts*baseline).
%               If (baseline == polynomial order), wts contains the polynomial
%               coefficients. Each row of (wts) can be used with the
%               "baseline" output (see below) to obtain the baseline
%               removed from the corresponding row of data.
%               Note that wts can also be used with the POLYVAL function to
%               reconstruct the baseline, however, a normalization factor
%               of 1./sqrt(n) must be used on each baseline to correct for
%               the number of variables (n).
%    baseline = Baseline used for each spectrum. Note: this is
%               the input (baseline) or polynomial basis.
%
%I/O: [bldata,wts,baseline] = wlsbaseline(data,baseline,options);
%I/O: [bldata,wts,baseline] = wlsbaseline(data,order,options);
%
%See also: BASELINE, BASELINEW, MSCORR, SAVGOL

%Copyright Eigenvector Research, Inc. 2004-2010
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.
%jms 9/20/04 added output of baseline basis
%jms 10/24/05 initialize with nonneg (if requested)

if nargin == 0;  data = 'io'; end
if isstr(data);
  options = [];

  %display and general settings
  options.plots   = 'none';
  options.nonneg  = 'no';

  options.weightmode = 1;  %EXPERIMENTAL   weightmode : [{1}| 2 ] method to handle weighting.
  
  %settings for weightmode 1
  options.power  = 2;
  options.negw   = 1;
  
  %settings for weightmode 2
  options.tsqlim   = 0.99;
  options.trbflag = 'bottom';

  %ending criteria
  options.delta  = 1e-9; %minimum change in fit
  options.maxiter = 100; %maximum iterations
  options.maxtime = 600; %maximum time in seconds

  options.definitions   = @optiondefs;

  if nargout == 0; evriio(mfilename,data,options); clear data; else; data = evriio(mfilename,data,options); end
  return
end

if nargin < 3; options = []; end
if nargin < 2; order = 2; end

options = reconopts(options,'wlsbaseline');
options.nonneg = strcmp(lower(options.nonneg),'yes');  %translate string to number (makes test in loop faster)

%determine if we're fitting the top or the bottom of the data
if strcmp(options.trbflag,'top');
  st = -1;
else
  st = 1;
end

%extract if was DSO
[m,n] = size(data);
if isa(data,'dataset');
  origdata = data;
  wasdso   = true;
  incl  = data.include{2};
  xaxis = data.axisscale{2};
  if ~isempty(xaxis)
    xaxis = xaxis(incl);
  else
    xaxis = incl;  %no axisscale? use include itself!
  end
  data  = data.data(:,incl);
  [m,n] = size(data);
else
  wasdso = false;
  incl  = 1:n;
  xaxis = 1:n;
end

if prod(size(order))==1;  %scalar? it actually IS order
  a = [];
  for i = 0:order
    a(i+1,:) = xaxis.^i;
  end
  a = normaliz(a,[],2);
else
  if size(order,2)>n & max(incl)<=size(order,2);  %is order TOO long but include appears to apply
    order = order(:,incl);  %apply include to the columns
  end
  if size(order,2)~=n;
    error('provided background does not match size of data');
  end
  a = order;   %use it as the baseline
  order = size(order,1)-1;
end  
b = ones(1,order+1);

if strcmp(options.plots,'final');
  subplot(2,1,1); plot(data); 
  ylabel('Original Data');
end

starttime = now;
tsqst    = ttestp(1-options.tsqlim,5000,2);
for specind = 1:m;
  spec = data(specind,:);
  
  res  = ones(size(spec));
  wts  = res;
  ores = res*inf;
  ossres = inf;
  negs = logical(ones(1,length(wts)));
  
  %initial guess as stats for weighting
  if options.nonneg;
    b = fastnnls(a',spec',0)';
  else
    b        = spec/a;
  end
  res      = st*(spec-b*a);
  reslimit = 0.05*rmse(res);
  for loop = 1:options.maxiter;
    
    switch options.weightmode
      case 1
        negs       = res<0;
        res(negs)  = real(res(negs)/(10.^options.negw));   %weight negative residuals
        res        = abs(res.^options.power);
        mres       = median(res);
        if mres==0;
          mres = 1;
        end
        wts        = (1+res./mres);

      case 2  
        tsq        = res/reslimit;
        negs       = tsq>tsqst;
        wts        = tsq.*0+1;
        wts(negs)  = (0.5 + tsq(negs)./tsqst);
        
    end
    
    ores = res;
    ob   = b;
    if options.nonneg;
      b = fastnnls((a./(ones(order+1,1)*wts))',(spec./wts)',0,b')';
    else
      b = (spec./wts)/(a./(ones(order+1,1)*wts));
    end        
    
    res  = st*(spec-b*a);
    ssres = rmse(res/reslimit);
    Dres = ssres-ossres;
    ossres = ssres;
    
    if strcmp(options.plots,'debug');
      subplot(3,1,1); 
      plot(xaxis,res);
      hline
      ylabel('Residuals');
      title(['Iterations ' num2str(loop) '  Delta Residuals: ' num2str(Dres)]);
      subplot(3,1,2);
      plot(xaxis,wts); 
      ylabel('Deweighting');
      subplot(3,1,3); 
      plot(xaxis,spec,1:n,b*a)
      ylabel('Orig and Baseline');
      drawnow
      pause;
    end
    
    if abs(Dres) < options.delta; break; end
    if (now-starttime)*60*60*24 > options.maxtime; break; end
    
  end
  
  if strcmp(options.plots,'intermediate');
    subplot(2,1,1); 
    plot(xaxis,res);
    hline
    ylabel('Residuals');
    title(['Iterations ' num2str(loop)]);
    subplot(2,1,2); 
    plot(xaxis,spec,xaxis,b*a)
    ylabel('Orig and Baseline');
    title(sprintf('Neg. Weight: %g   Resid. Order: %g',options.negw,options.power))
    drawnow
%     pause
  end
  
  if nargout >= 2;
    allb(specind,:) = b;
  end
  data(specind,:) = spec-b*a;
end

if strcmp(options.plots,'final');
  plot(xaxis,data)
  ylabel('Baselined Data');
  title(sprintf('Neg. Weight: %g   Resid. Order: %g',options.negw,options.power))
  drawnow
end

if wasdso
  %insert baselined data back into DSO object
  origdata.data = origdata.data.*nan;
  origdata.data(:,incl) = data;
  data = origdata;
end

%--------------------------
function out = optiondefs

defs = {
  %name                    tab           datatype        valid                            userlevel       description
  'plots'                  'Display'     'select'        {'none' 'final' 'intermediate' 'debug'} 'novice'        'Governs plotting. Final returns a final plot at end of run, Intermediate returns plots after each spectrum is completed, Debug plots multiple steps during analyses, None gives no plots.'
  'weightmode'             'Algorithm'    'select'        {1 2}                            'intermediate'  'Weighting mode to use: Mode 1 = Power method. Negative residuals are weighted up by the power of 10.^(option.negw). All residuals are then raised to the power of (option.power). Mode 2 = T squared method. Negative residuals are weighted up by the extent to which the surpass an estimate of the noise limit and the approximate t-limit defined by (option.tsqlim).'
  'negw'                   'Algorithm'    'double'        []                               'advanced'  'Upweighting scale of negative values (10^negw) (used only for weightmode = 1)';
  'power'                  'Algorithm'    'double'        []                               'advanced'  'Exponential amplification of residuals (used only for weightmode = 1)';
  'tsqlim'                 'Algorithm'    'double'        []                               'intermediate'  'T-test confidence limit for significant negative residuals which need to be up-weighted. (used only for weightmode = 2)';
  'trbflag'                'Algorithm'    'select'        {'bottom' 'top'}                 'intermediate'  'Baseline to: top or bottom of data.';
  'nonneg'                 'Algorithm'    'select'        {'no' 'yes'}                     'advanced'  'Flag to force non-negative baseline weighting. Most often used when "real" spectra are used for baslineing and they should not be "flipped" by a negative weighting.';
  'delta'                  'End Criteria'    'double'        []                          'intermediate'  'Change-of-fit convergence criterion.';
  'maxiter'                'End Criteria'    'double'        []                          'intermediate'  'Maximum iterations allowed per spectrum.';
  'maxtime'                'End Criteria'    'double'        []                          'intermediate'  'Maximum time (in seconds) permitted for baselining of all data.';
};
out = makesubops(defs);
