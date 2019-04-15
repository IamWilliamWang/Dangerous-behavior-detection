import cv2
import numpy as np


class FileUtil:
    def CutVideo(oldVideoFilename, newVideoFilename, fromFrame, toFrame):
        '''
        @Deprecated 剪切视频，储存到新文件中
        :param oldVideoFilename:
        :param newVideoFilename:
        :param fromFrame:
        :param toFrame:
        :return:
        '''
        # 打开输入输出视频文件
        videoInput = cv2.VideoCapture(oldVideoFilename)
        # 获得码率及尺寸
        fps = videoInput.get(cv2.CAP_PROP_FPS)
        size = (int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frameCount = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)
        if toFrame > frameCount:
            return
        videoOutput = cv2.VideoWriter(newVideoFilename, cv2.VideoWriter_fourcc(*'DIVX'), fps, size, False)  # MPEG-4编码
        framesLen = toFrame - fromFrame
        videoInput.set(cv2.CAP_PROP_POS_FRAMES, fromFrame)
        while videoInput.isOpened() and framesLen >= 0:
            ret, frame = videoInput.read()
            if ret is False:
                break
            videoOutput.write(frame)
            framesLen -= 1
        videoInput.release()
        videoOutput.release()

    def OpenVideos(inputVideoFilename=None, outputVideoFilename=None, outputVideoEncoding='DIVX'):  # MPEG-4编码
        '''
        打开输入输出视频文件
        :param outputVideoFilename:
        :param outputVideoEncoding:
        :return:
        '''
        videoInput = None
        videoOutput = None
        if inputVideoFilename is not None:
            videoInput = FileUtil.OpenInputVideo(inputVideoFilename)  # 打开输入视频文件
        if outputVideoFilename is not None:
            videoOutput = FileUtil.OpenOutputVideo(outputVideoFilename, videoInput, outputVideoEncoding)
        return videoInput, videoOutput

    def OpenInputVideo(inputVideoFilename):
        '''
        打开输入视频文件
        :return:
        '''
        return cv2.VideoCapture(inputVideoFilename)

    def OpenOutputVideo(outputVideoFilename, inputFileStream, outputVideoEncoding='DIVX'):
        '''
        打开输出视频文件
        :param inputFileStream:
        :param outputVideoEncoding:
        :return:
        '''
        # 获得码率及尺寸
        fps = int(inputFileStream.get(cv2.CAP_PROP_FPS))
        size = (int(inputFileStream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(inputFileStream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cv2.VideoWriter(outputVideoFilename, cv2.VideoWriter_fourcc(*outputVideoEncoding), fps, size,
                               False)

    def CloseVideos(inputVideoStream=None, outputVideoStream=None):
        '''
        关闭输入输出视频文件
        :param outputVideoStream:
        :return:
        '''
        if inputVideoStream is not None:
            inputVideoStream.release()
        if outputVideoStream is not None:
            outputVideoStream.release()


class Transformer:
    def Imread(filename_unicode):
        '''
        读取含有unicode文件名的图片
        :return:
        '''
        return cv2.imdecode(np.fromfile(filename_unicode, dtype=np.uint8), -1)

    def BGR2Gray(BGR):
        '''
        将读取的BGR转换为单通道灰度图
        :param BGR: BGR图片
        :return: 灰度图
        '''
        return cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)

    def Gray2Edges(grayFrame):
        '''
        将灰度图调用canny检测出edges，返回灰度edges图
        :param grayFrame: 灰度图
        :return: 含有各个edges的黑白线条图
        '''
        grayFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)  # 高斯模糊，去除图像中不必要的细节
        edges = cv2.Canny(grayFrame, 50, 150, apertureSize=3)
        return edges

    def GetEdgesFromImage(BGRimage):
        '''
        将彩色图转变为带有所有edges信息的黑白线条图
        :return:
        '''
        return Transformer.Gray2Edges(Transformer.BGR2Gray(BGRimage))

    def GetLinesFromGrayEdges(grayImg, threshold=200):
        '''
        单通道灰度图中识别内部所有线段并返回
        :param grayImg: 灰度图
        :param threshold: 阈值限定，线段越明显阈值越大。小于该阈值的线段将被剔除
        :return:
        '''
        return cv2.HoughLines(grayImg, 1, np.pi / 180, threshold)


class PlotUtil:
    def WriteLinesOnImage(img, lines, lineCount=1):
        '''
        在图img中划lineCount条线，线段的优先级由长到短
        :param img: BGR图片
        :param lines: HoughLines返回的变量
        :param lineCount: 要画线的个数
        :return:
        '''
        for i in range(lineCount):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    def PutText(img, text):
        cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


def Otsu_canny(image, lowrate=0.1):
    '''
    @Deprecated 自动选择参数，调用canny算子。
    :param image:
    :param lowrate:
    :return:
    '''
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged


def LinesEquals(lines1, lines2, compareLineCount):
    '''
    HoughLines返回的lines判断是否相等
    :param lines1: 第一个lines
    :param lines2: 第二个lines
    :param compareLineCount: 比较前几条line
    :return: 是否二者相等
    '''
    sameCount = 0
    diffCount = 0
    try:
        for i in range(compareLineCount):
            for rho1, theta1 in lines1[i]:
                for rho2, theta2 in lines2[i]:
                    if rho1 != rho2 or theta1 != theta2:
                        diffCount += 1
                    else:
                        sameCount += 1
    except IndexError:  # 阈值过高的话会导致找不到那么多条line
        pass
    return sameCount / (sameCount + diffCount) > 0.9  # 不同到一定程度再报警

def EdgesLinesEquals(edges, compareLineCount):
    for i in range(len(edges)-1):
        for j in range(i+1, len(edges),2):
            if LinesEquals(edges[i],edges[j],compareLineCount):
                return True
    return False

FirstFramePosition = None
LastFramePosition = None


def GetStaticThings_fromVideo(videoFilename, startFrameRate=0., endFrameRate=1., outputEdgesFilename=None):
    '''
    从视频文件中提取不动物体的帧
    :param videoFilename: 文件名
    :param startFrameRate 开始读取帧处于视频的比例，必须取0-1之间
    :param outputEdgesFilename （测试用）EdgesFrame全部输出到视频为该名的文件中
    :return: 不动物体的Edges帧
    '''
    # 打开输入输出视频文件
    videoInput = FileUtil.OpenInputVideo(videoFilename)
    frame_count = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)
    outputVideo = None
    if outputEdgesFilename is not None:
        outputVideo = FileUtil.OpenOutputVideo(outputEdgesFilename, videoInput)
    staticEdges = None
    videoInput.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * startFrameRate))  # 指定读取的开始位置
    global FirstFramePosition  # 记录第一帧的位置
    global LastFramePosition
    FirstFramePosition = int(frame_count * startFrameRate)
    LastFramePosition = int(frame_count * endFrameRate)
    if endFrameRate != 1:
        frame_count = int(frame_count * (endFrameRate - startFrameRate))
    while videoInput.isOpened() and frame_count >= 0:
        ret, frame = videoInput.read()
        if ret is False:
            break
        edgeFrame = Transformer.GetEdgesFromImage(frame)  # 边缘识别
        if staticEdges is None:
            staticEdges = edgeFrame  # 初始化staticBW
        else:
            staticEdges &= edgeFrame  # 做与运算，不同点会被去掉
        if outputEdgesFilename is not None:
            outputVideo.write(edgeFrame)  # 写入边缘识别结果
        frame_count -= 1
        # cv2.imshow('frame', edgeFrame)
        # if cv2.waitKey(2) & 0xFF == ord('q'):
        #   break

        FileUtil.CloseVideos(videoInput, outputVideo)
    return staticEdges


def GetStaticEdges_fromSteam(inputStream, outputEdgesFilename=None):
    '''
    从输入流中提取不动物体的Edges帧
    :param inputStream: 输入文件流
    :param startFrameRate 开始读取帧处于视频的比例，必须取0-1之间
    :param outputEdgesFilename （测试用）EdgesFrame全部输出到视频为该名的文件中
    :return: 不动物体的Edges帧
    '''
    # 打开输入输出视频文件
    videoInput = inputStream
    outputVideo = None
    if outputEdgesFilename is not None:
        outputVideo = FileUtil.OpenOutputVideo(outputEdgesFilename, videoInput)
    staticEdges = None
    frame_count = 20  # 截取20帧
    while videoInput.isOpened() and frame_count >= 0:
        ret, frame = videoInput.read()
        if ret is False:
            break
        edgeFrame = Transformer.GetEdgesFromImage(frame)  # 边缘识别
        if staticEdges is None:
            staticEdges = edgeFrame  # 初始化staticBW
        else:
            staticEdges &= edgeFrame  # 做与运算，不同点会被去掉
        if outputEdgesFilename is not None:
            outputVideo.write(edgeFrame)  # 写入边缘识别结果
        frame_count -= 1
    return staticEdges


def main_FileStream():
    videoFilename = '开关柜3.mp4'
    staticLines = None  # 储存基准帧
    # 初始化处理参数
    compareLineCount = 3  # 比较几条线
    videoClipCount = 26  # 10、26
    # 开始生成静态基准并进行检测
    for segmentIndex in range(0, videoClipCount):
        segmentRate = 1 / videoClipCount
        videoInput = cv2.VideoCapture(videoFilename)
        videoFps = int(videoInput.get(cv2.CAP_PROP_FPS))
        staticThings = GetStaticThings_fromVideo(videoFilename, startFrameRate=segmentRate * segmentIndex,
                                                 endFrameRate=segmentRate * (segmentIndex + 1),
                                                 outputEdgesFilename=None)  # 获得不动的物体
        error = False
        lines = Transformer.GetLinesFromGrayEdges(staticThings, threshold=50)
        if staticLines is None:
            staticLines = lines  # 以第一段视频检测出的线为基准（因为第一段视频没有人）
        else:
            frameCount = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)
            startFrameIndex = int(segmentIndex * segmentRate * frameCount)
            endFrameIndex = int((segmentIndex + 1) * segmentRate * frameCount) - 1
            if LinesEquals(lines, staticLines, compareLineCount):
                print('未检测到异常。', startFrameIndex / videoFps, '-', endFrameIndex / videoFps, '秒', sep='')
            else:
                # CutVideo(videoFilename, 'ExceptionVideo' + str(startFrameIndex) + '.mp4', startFrameIndex, endFrameIndex)
                print('检测到异常！！', startFrameIndex / videoFps, '-', endFrameIndex / videoFps, '秒', sep='')
                error = True

        # 获得检测线条的视频片段第一帧
        videoInput.set(cv2.CAP_PROP_POS_FRAMES, FirstFramePosition)
        for i in range(FirstFramePosition, LastFramePosition):
            _, frame = videoInput.read()
            # 向这帧图像画线
            # WriteLinesOnImage(frame, lines, compareLineCount)
            if error:
                PlotUtil.PutText(frame, 'Warning')

            cv2.imshow('result', frame)
            if cv2.waitKey(1) is 27:  # Esc按下
                break
            # cv2.destroyAllWindows()


def main_VideoStream(source='rtsp://admin:1234abcd@192.168.1.64'):
    # 初始化输入流
    # 获得静态Edges的Lines信息
    inputStream = cv2.VideoCapture(source)
    staticEdges = GetStaticEdges_fromSteam(inputStream)
    staticLines = Transformer.GetLinesFromGrayEdges(staticEdges)
    # 初始化处理参数
    compareLineCount = 2
    errorTimes = 0
    showWarning = False
    containedLines = []
    # 启动检测
    while inputStream.isOpened():
        # Capture frame-by-frame  
        ret, frame = inputStream.read()
        if ret is False:
            break
        if len(containedLines) < 5:
            lines = Transformer.GetLinesFromGrayEdges(Transformer.GetEdgesFromImage(frame), threshold=50)
            containedLines += [lines]
            continue
        if EdgesLinesEquals(containedLines, compareLineCount):
            print('未检测到异常。')
            errorTimes -= 1
        else:
            print('检测到异常！！')
            errorTimes += 1
        # 清理保存的容器
        containedLines = []
        # 向这帧图像画线
        PlotUtil.WriteLinesOnImage(frame, lines, compareLineCount)
        # Display the resulting frame  
        if errorTimes > 1:
            showWarning = True
        if showWarning:
            PlotUtil.PutText(frame, 'Warning')
        if errorTimes <= 1:
            showWarning = False
        cv2.imshow('result', frame)
        if cv2.waitKey(1) is 27:  # Esc按下
            break
    # When everything done, release the capture  
    inputStream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main_VideoStream()
