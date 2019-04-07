% edgesFrame = getEdgesFromVideo('���ع�_��Ե���.avi');
% [~,~,~,frameCount] = size(edgesFrame);
% startFrame=frameCount*3/4; %����Ӷ���֡����β
% BW=getStaticFrame(edgesFrame,startFrame);
load('staticBW','BW');

[H,T,R]=hough(BW);
peaks=houghpeaks(H,2);
lines = houghlines(BW,T,R,peaks);%�û���任����������
%����
for k = 1:length(lines)
    p1 = lines(k).point1;
    p2 = lines(k).point2;
    plot(p1,p2,'LineWidth',4)
end
 %�ҷ�ֱ�߼�������
[H2,T2,R2] = hough(BW,'Theta',-75:0.1:-20);
Peaks1=houghpeaks(H2,5);
lines1=houghlines(Iedge,T2,R2,Peaks1);
for k=1:length(lines1)
xy1=[lines1(k).point1;lines1(k).point2];   
plot(xy1(:,1),xy1(:,2),'LineWidth',4);
end

hold off