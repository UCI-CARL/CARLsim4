% Version 12/8/2014
addpath ../../../tools/offline_analysis_toolbox

list={'R','M','B','C','G','Y'};
R=struct;
colors = [0:255,0:255,zeros(1,256); 255:-1:0,zeros(1,256),0:255; zeros(1,256),255:-1:0,255:-1:0]/255;
frameDurMs=100;
for j=1:length(list)
	SR = SpikeReader(['../results/spkV4' list{j} '.dat']);
	spkData = SR.readSpikes(frameDurMs);
    if size(spkData,1) < size(colors,2), spkData(size(colors,2),1) = 0; end
    
    R.(list{j})=mean(spkData,2)*1000/frameDurMs;
	clear SR;
end

figure,
for j=1:length(list), 
    ind = j;
    if j>3, ind = 10-j; end
    subplot(2,3,ind);
    for i=1:size(colors,2)
        plot3(colors(1,i),colors(2,i),R.(list{j})(i),'.','color',colors(:,i)); hold on;
    end
    title(list{j});
    set(gca,'zlim',[0 40]);
    zlabel('Hz');
end;
