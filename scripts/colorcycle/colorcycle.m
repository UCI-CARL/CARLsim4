list={'R','M','B','C','G','Y'};
R=struct;
colors = [0:255,0:255,zeros(1,256); 255:-1:0,zeros(1,256),0:255; zeros(1,256),255:-1:0,255:-1:0]/255;
FrameDur=100;
for j=1:length(list)
    s = readSpikes(['../../results/colorcycle/spkV4' list{j} '.dat'],FrameDur);
    
    if size(s,1) < size(colors,2), s(size(colors,2),1) = 0; end
    
    R.(list{j})=mean(s,2)*1000/FrameDur;
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
