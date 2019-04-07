% edgesFrame = getEdgesFromVideo('开关柜_边缘输出.avi');
% [~,~,~,frameCount] = size(edgesFrame);
% startFrame=frameCount*3/4; %处理从多少帧到结尾
% BW=getStaticFrame(edgesFrame,startFrame);
load('staticBW','BW');

[H,T,R]=hough(BW);
peaks=houghpeaks(H,2);
lines = houghlines(BW,T,R,peaks);%用霍夫变换检测和连接线
%画线
for k = 1:length(lines)
    p1 = lines(k).point1;
    p2 = lines(k).point2;
    plot(p1,p2,'LineWidth',4)
end
 %右方直线检测与绘制
[H2,T2,R2] = hough(BW,'Theta',-75:0.1:-20);
Peaks1=houghpeaks(H2,5);
lines1=houghlines(Iedge,T2,R2,Peaks1);
for k=1:length(lines1)
xy1=[lines1(k).point1;lines1(k).point2];   
plot(xy1(:,1),xy1(:,2),'LineWidth',4);
end

hold off