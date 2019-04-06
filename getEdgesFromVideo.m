function [outFrames] = getEdgesFromVideo(videoName)
    tic; %参数处理
    if nargin<1
        videoName = '开关柜.mp4';
    end
    %读取视频
    video = VideoReader(videoName);
    frames=[]; %储存所有的帧
    frames=read(video);
    [~,~,~,frameCount]=size(frames);
    grayFrames=[];
    outFrames=[];
    %遍历所有帧
    for i=1:frameCount
        I=rgb2gray(frames(:,:,:,i)); %灰度处理
        BW=edge(I,'Canny');
        grayFrames(:,:,:,i)=I;
        outFrames(:,:,:,i)=BW;
    end
    disp(strcat('Total time: ',num2str(toc),' s'))
end