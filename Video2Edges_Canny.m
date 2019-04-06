function Video2Edges(videoName,outputName,showDetail)
    tic; %参数处理
    if nargin<3
        showDetail = '1';
    end
    if nargin<2
        outputName = '输出';
    end
    if nargin<1
        videoName = '开关柜.mp4';
    end
    %读取视频
    video = VideoReader(videoName);
    frames=[]; %储存所有的帧
    % for currTime=0:video.FrameRate
    %     if hasFrame(video) == false
    %         break;
    %     end
    %     %video.CurrentTime=currTime;
    %     frame = readFrame(video);
    %     frames(:,:,:,currTime+1)=frame;
    % end
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
    %     imwrite(I,strcat(mat2str(i),'.png'));
    %     imwrite(BW,strcat('0',mat2str(i),'.png'));
    end
    % implay(outFrames)
    % videoOut = VideoWriter('gray.mp4');
    % open(videoOut);
    % writeVideo(videoOut,grayFrames);
    % close(videoOut);
    videoOut = VideoWriter(outputName);
    open(videoOut);
    writeVideo(videoOut,outFrames);
    close(videoOut);
    if showDetail ~= '0'
        disp(strcat('Total time: ',num2str(toc),' s'))
    end
end