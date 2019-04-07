edgesFrame = getEdgesFromVideo('开关柜_边缘输出.avi');
[~,~,~,frameCount] = size(edgesFrame);
for loop=1:4
    startFrame=frameCount*loop/(loop+1); %处理从多少帧到结尾
    result=getStaticFrame(edgesFrame,startFrame);
    subplot(2,2,loop)
    imshow(result)
    title(strcat(mat2str(loop), '/', mat2str(loop+1) ) )
end