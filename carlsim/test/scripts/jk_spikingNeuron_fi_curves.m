% function takes neuron type as input
% Author: Jeff Krichmar
% Modified: Hirak Kashyap -- add LIF refractory
%   RS - regular spiking
%   Class1 - Class 1 firing behavior
%   Class2 - Class 2 firing behavior
%   IF - leaky integrate and fire
function jk_spikingNeuron_fi_curves (type)


% set the current injection range based on the neuron type.
if strcmp (type, 'IF')
    %i_inj = 0:0.04:4;
    i_inj = 0:0.8:8;
elseif strcmp (type, 'Class1')
    i_inj = 0:0.5:50;
elseif strcmp (type, 'Class2')
    i_inj = 0:0.05:5;
else
    i_inj = 0:0.1:10;
end

f = zeros(1,size(i_inj,2));

% ramp through the currents
for i=1:size(i_inj,2)
    
    % at about the mid-range of the currents plot the voltage traces
    if i == round(size(i_inj,2)/4)
        subplot(121)
        f(i) = izzyrate(type,i_inj(i),1);
        title([type, ' Neuron, Iinj = ', num2str(i_inj(i))])
    else
        f(i) = izzyrate(type,i_inj(i),0);
    end
end
f=f/10;
subplot(122)
plot(i_inj, f, '*')
title([type, ' Neuron'])
xlabel ('Injection Current (I)')
ylabel ('Firing Rate (Hz)')
disp(i_inj);disp(f);
end

% returns firing rate as a function of
%   type - neuron type
%   Iinj - injection current
%   bplot - plot voltage trace if true
function fr = izzyrate (type, Iinj, bplot)

% set neuron model parameters based on type
if strcmp ('RS', type)
    a = 0.02;
    b = 0.2;
    c = -65;
    d = 8;
    thr = 30;
    v = c;
    u = 0.2.*v;
elseif strcmp ('Class1', type)
    a = 0.02;
    b = -0.1;
    c = -55;
    d = 6;
    thr = 30;
    v = c;
    u = 0.2.*v;
elseif strcmp ('Class2', type)
    a = 0.2;
    b = 0.26;
    c = -65;
    d = 0;
    thr = 30;
    v = c;
    u = 0.2.*v;
elseif strcmp ('IF', type)
    EL = -65;
    Vreset = EL;
    thr = -50;
    tm = 10;
    Rm = 5;
    v = Vreset;
    
%     EL = 0;
%     Vreset = EL;
%     thr = 1;
%     tm = 20;
%     Rm = 1.5;
%     v = Vreset;
    
    abs_ref = 2;
    c_ref = 0;
else
    disp (['ERROR: bad cell type: ', type])
end

% Do not set the injection current for the first 100 milliseconds
I = 0;
for t = -100:10000
    if ~strcmp('IF', type)
        [u,v] = izzy(u, v, a, b, c, d, thr, I);
    else
        [v, c_ref] = lif(v, EL, Vreset, thr, tm, Rm, abs_ref, c_ref, I);
    end
    
    if t > 0
        I = Iinj;
        if v > thr
            Em(t) = 10;
        else
            Em(t) = v;
        end
    end
end

% use the find peaks function to allow the models to settle at steady state
fr = sum(findpeaks(Em) > 0);

if bplot
    plot (Em)
end
end

% Izhikevich simple neuron. Advances u and v one timestep. Assumes time
% step is 1 milliseconds
function [uout, vout] = izzy(uin, vin, a, b, c, d, thr, I)

if vin > thr
    vout = c;
    uout = uin + d;
else
    % use forward Euler numerical method.
    vout=vin+0.5*((0.04*vin+5).*vin+140-uin+I);    % for numerical
    vout=vout+0.5*((0.04*vout+5).*vout+140-uin+I);    % stability time
    uout=uin+a.*(b*vout-uin);                   % step is 0.5 ms
end
end

% Leaky integrate and fire neuron. Advances v one timestep. Assumes time
% step is 1 millisecond.
function [vout, c_ref] = lif(vin, EL, Vreset, thr, tm, Rm, abs_ref, c_ref, I)
if c_ref > 0
    c_ref = c_ref - 1;
    vout = Vreset;
else
    if vin > thr
        vout = Vreset;
        c_ref = abs_ref;
    else
        vout = vin + 1/tm * (EL - vin + Rm*I);
    end
end
end

