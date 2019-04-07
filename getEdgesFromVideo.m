function [edgeFrames] = getEdgesFromVideo(videoName)
% 从视频文件中读取所有edges(Canny)并返回四维数组结果
% 四位数组为[帧宽,帧高,帧深度=1,总帧数]
% 输入参数：[videoName]文件名
    video = VideoReader(videoName);%读取视频
    frames=[]; %储存所有的帧
    frames=read(video);
    [~,~,~,frameCount]=size(frames);
    %grayFrames=[];
    edgeFrames=[];
    %遍历所有帧
    for i=1:frameCount
        I=rgb2gray(frames(:,:,:,i)); %灰度处理
        BW=edge(I,'Canny');
        %grayFrames(:,:,:,i)=I;
        edgeFrames(:,:,:,i)=BW;
    end
end