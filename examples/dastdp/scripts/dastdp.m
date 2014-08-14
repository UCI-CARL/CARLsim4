% Version 8/13/14

addpath('../results');
weights = csvread('weight.csv');
[n,x] = hist(weights);
bar(x,n,1);
idx = abs(x - weights(11)) < 0.0051; % find the bin containing weights(11)
hold on;
bar([x(idx) - 0.01,x(idx),x(idx) + 0.01],[0,n(idx),0],1,'r'); % plot the red bar