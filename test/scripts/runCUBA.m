function fired = runCUBA(runMs, rate, wt)

if nargin<3,wt=15;end
if nargin<2,rate=10;end
if nargin<1,runMs=1000;end

a = 0.02;
b = 0.2;
c = -65;
d = 8.0;

isi = 1000/rate;

v = zeros(1,runMs);
u = zeros(1,runMs);
I = zeros(1,runMs);

v(1) = c;
u(1) = b*v(1);

fired = [];
for t=1:runMs
    if mod(t-1,isi)==0
        I(t) = wt;
    end
    
    if t==1
        continue;
    end
    
    vv = v(t-1);
    uu = u(t-1);

    if vv>=30
        vv = c;
        uu = uu + d;
        fired = [fired; t, 1];
    end
    
    vv=vv+0.5*(0.04*vv.^2+5*vv+140-uu+I(t-1)); % step 0.5 ms
    vv=vv+0.5*(0.04*vv.^2+5*vv+140-uu+I(t-1)); % for numerical
    v(t) = vv;
    u(t) = uu + a*(b*vv-uu);
end

v(v>=30) = 30; % for plotting

subplot(221)
plot(v)
title('voltage')

subplot(222)
plot(u)
title('recovery')

subplot(223)
plot(I)
title('current')

subplot(224)
if numel(fired)>1
    plot(fired(:,1), fired(:,2), '.');
    axis([0 runMs 0 2])
end
title('raster plot')

disp(['firing rate = ' num2str(size(fired,1)/runMs*1000.0)]);
end

