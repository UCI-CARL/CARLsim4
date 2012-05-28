function s = readspikes(file, FrameDur)

fid=fopen(file,'r');
nrRead=1000000;
d2=zeros(0,nrRead);
s=0;
i=0;

while size(d2,2)==nrRead
    d2=fread(fid,[2 nrRead],'uint32');
    d=d2;
    if ~isempty(d)
        if size(s,2)~=max(d(2,:))+1 || size(s,1)~=floor(d(1,end)/(FrameDur))+1, s(floor(d(1,end)/(FrameDur))+1,max(d(2,:))+1)=0; end
        s=s+full(sparse(floor(d(1,:)/(FrameDur))+1,d(2,:)+1,1,size(s,1),size(s,2)));
    end
    i=i+1;
%      if i==2,break;end;
end
fclose(fid);