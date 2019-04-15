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
    '''
    图像转换器，负责图像的读取，变灰度，边缘检测和线段识别。
    《请遵守以下命名规范：前缀image、img代表彩色图。前缀为gray代表灰度图。前缀为edges代表含有edge的黑白图。前缀为lines代表edges中各个线段的结构体。前缀为static代表之后的比较要以该变量为基准进行比较。》
    '''
    def Imread(filename_unicode):
        '''
        读取含有unicode文件名的图片
        :return:
        '''
        return cv2.imdecode(np.fromfile(filename_unicode, dtype=np.uint8), -1)

    def BGR2Gray(image):
        '''
        将读取的BGR转换为单通道灰度图
        :param image: BGR图片
        :return: 灰度图
        '''
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def Gray2Edges(grayFrame):
        '''
        将灰度图调用canny检测出edges，返回灰度edges图
        :param grayFrame: 灰度图
        :return: 含有各个edges的黑白线条图
        '''
        grayFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)  # 高斯模糊，去除图像中不必要的细节
        edges = cv2.Canny(grayFrame, 50, 150, apertureSize=3)
        return edges

    def GetEdgesFromImage(imageBGR):
        '''
        将彩色图转变为带有所有edges信息的黑白线条图
        :param imageBGR: 彩色图
        :return:
        '''
        return Transformer.Gray2Edges(Transformer.BGR2Gray(imageBGR))

    def GetLinesFromEdges(edgesFrame, threshold=200):
        '''
        单通道灰度图中识别内部所有线段并返回
        :param edgesFrame: edges图
        :param threshold: 阈值限定，线段越明显阈值越大。小于该阈值的线段将被剔除
        :return:
        '''
        return cv2.HoughLines(edgesFrame, 1, np.pi / 180, threshold)


class PlotUtil:
    '''
    用于显示图片的帮助类。可以在彩图中画霍夫线
    '''
    def PaintLinesOnImage(img, houghLines, paintLineCount=1):
        '''
        在彩色图中划指定条霍夫线，线段的优先级由长到短
        :param img: BGR图片
        :param houghLines: 霍夫线，即HoughLines函数返回的变量
        :param paintLineCount: 要画线的个数
        :return:
        '''
        for i in range(paintLineCount):
            for rho, theta in houghLines[i]:
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
        '''
        在彩图img上使用默认字体写字
        :param text: 要写上去的字
        :return:
        '''
        cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


def LinesEquals(lines1, lines2, comparedLinesCount):
    '''
    HoughLines函数返回的lines判断是否相等
    :param lines1: 第一个lines
    :param lines2: 第二个lines
    :param comparedLinesCount: 比较前几条line
    :return: 是否二者相等
    '''
    sameCount = 0
    diffCount = 0
    try:
        for i in range(comparedLinesCount):
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
    for i in range(len(edges) - 1):
        for j in range(i + 1, len(edges), 2):
            if LinesEquals(edges[i], edges[j], compareLineCount):
                return True
    return False


FirstFramePosition = None
LastFramePosition = None


def GetStaticEdges_fromVideo(videoFilename, startFrameRate=0., endFrameRate=1., outputEdgesFilename=None):
    '''
    从视频文件中提取不动物体的帧
    :param videoFilename: 文件名
    :param startFrameRate 开始读取帧处于视频的比例，必须取0-1之间
    :param outputEdgesFilename （测试用）EdgesFrame全部输出到视频为该名的文件中
    :return: 不动物体的Edges帧
    '''
    # 打开输入输出视频文件
    videoInput = FileUtil.OpenInputVideo(videoFilename)
    frame_count = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总共的帧数
    outputVideo = None  # 声明输出文件
    if outputEdgesFilename is not None:
        outputVideo = FileUtil.OpenOutputVideo(outputEdgesFilename, videoInput)
    staticEdges = None  # 储存固定的Edges
    videoInput.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * startFrameRate))  # 指定读取的开始位置
    global FirstFramePosition  # 记录第一帧的位置
    global LastFramePosition  # 记录最后一帧的位置
    FirstFramePosition = int(frame_count * startFrameRate)
    LastFramePosition = int(frame_count * endFrameRate)
    if endFrameRate != 1:  # 如果提前结束，则对总帧数进行修改
        frame_count = int(frame_count * (endFrameRate - startFrameRate))
    while videoInput.isOpened() and frame_count >= 0:  # 循环读取
        ret, frame = videoInput.read()
        if ret is False:
            break
        edges = Transformer.GetEdgesFromImage(frame)  # 对彩色帧进行边缘识别
        if staticEdges is None:
            staticEdges = edges  # 初始化staticEdges
        else:
            staticEdges &= edges  # 做与运算，不同点会被去掉
        if outputEdgesFilename is not None:
            outputVideo.write(edges)  # 写入边缘识别结果
        frame_count -= 1
        FileUtil.CloseVideos(videoInput, outputVideo)
    return staticEdges


def GetStaticEdges_fromSteam(inputStream, frame_count=20, outputEdgesFilename=None):
    '''
    从输入流中提取不动物体的Edges帧
    :param inputStream: 输入文件流
    :param frame_count 要读取的帧数
    :param outputEdgesFilename （测试用）EdgesFrame全部输出到视频为该名的文件中
    :return: 不动物体的Edges帧
    '''
    outputVideo = None
    if outputEdgesFilename is not None:
        outputVideo = FileUtil.OpenOutputVideo(outputEdgesFilename, inputStream)
    staticEdges = None
    while inputStream.isOpened() and frame_count >= 0:
        ret, frame = inputStream.read()
        if ret is False:
            break
        edges = Transformer.GetEdgesFromImage(frame)  # 边缘识别
        if staticEdges is None:
            staticEdges = edges  # 初始化staticBW
        else:
            staticEdges &= edges  # 做与运算，不同点会被去掉
        if outputEdgesFilename is not None:
            outputVideo.write(edges)  # 写入边缘识别结果
        frame_count -= 1
    return staticEdges


def main_FileStream(videoFilename = '开关柜3.mp4',compareLineCount = 3,videoClipCount = 26): # 2.mp4用10、3.mp4用26
    '''
    针对视频文件进行的开关柜检测主函数
    :param videoFilename: 视频文件名
    :param compareLineCount: 需要比较几条线是一样的
    :param videoClipCount: 视频要分成多少段
    :return:
    '''
    staticLines = None  # 储存基准帧
    # 开始生成静态基准并进行检测
    for segmentIndex in range(0, videoClipCount):
        segmentRate = 1 / videoClipCount  # 一小段是百分之多少
        videoInput = cv2.VideoCapture(videoFilename)  # 打开视频文件
        videoFps = int(videoInput.get(cv2.CAP_PROP_FPS))  # 读取Fps，取整
        staticThings = GetStaticEdges_fromVideo(videoFilename, startFrameRate=segmentRate * segmentIndex,
                                                endFrameRate=segmentRate * (segmentIndex + 1))  # 获得不动的物体
        error = False
        lines = Transformer.GetLinesFromEdges(staticThings, threshold=50)
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
    staticLines = Transformer.GetLinesFromEdges(staticEdges)
    # 初始化处理参数
    compareLineCount = 3
    errorTimes = 0
    showWarning = False
    nowStaticBW = None
    restFrames = 10
    # 启动检测
    while inputStream.isOpened():
        # Capture frame-by-frame
        ret, frame = inputStream.read()
        if ret is False:
            break
        if restFrames > 0:
            if nowStaticBW is None:
                pass
            continue
        lines = Transformer.GetLinesFromEdges(Transformer.GetEdgesFromImage(frame), threshold=50)
        if EdgesLinesEquals(containedLines, compareLineCount):
            print('未检测到异常。')
            errorTimes -= 1
        else:
            print('检测到异常！！')
            errorTimes += 1
        # 清理保存的容器
        containedLines = []
        # 向这帧图像画线
        PlotUtil.PaintLinesOnImage(frame, lines, compareLineCount)
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
    cv2.destroyAllWindows()
    inputStream.release()


if __name__ == '__main__':
    main_FileStream('开关柜3.mp4')
