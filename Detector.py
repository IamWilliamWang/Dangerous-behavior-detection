import cv2
import numpy as np

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

def BGR2Gray(BGR):
    '''
    将读取的BGR转换为单通道灰度图
    :param BGR: BGR图片
    :return: 灰度图
    '''
    return cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)

def Gray2Edge(grayFrame):
    '''
    将灰度图调用canny检测出edges，返回灰度edges图
    :param grayFrame: 灰度图
    :return: 含有各个edges的灰度图
    '''
    grayFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)  # 高斯模糊，去除图像中不必要的细节
    edges = cv2.Canny(grayFrame, 50, 150, apertureSize=3)
    return edges

def GetLines(grayImg, threshold=200):
    '''
    单通道灰度图中识别内部所有线段并返回
    :param grayImg: 灰度图
    :param threshold: 阈值限定，线段越明显阈值越大。小于该阈值的线段将被剔除
    :return:
    '''
    return cv2.HoughLines(grayImg, 1, np.pi / 180, threshold)

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

def GetStaticFrame_Edges(videoFilename, startFrameRate = 0, outputEdgesFilename=None):
    '''
    从视频文件中提取不动物体的Edges帧
    :param videoFilename: 文件名
    :param startFrameRate 开始读取帧处于视频的比例，必须取0-1之间
    :param outputEdgesFilename （测试用）EdgesFrame全部输出到视频为该名的文件中
    :return: 不动物体的Edges帧
    '''
    # 打开输入输出视频文件
    videoInput = cv2.VideoCapture(videoFilename)
    # 获得码率及尺寸
    fps = videoInput.get(cv2.CAP_PROP_FPS)
    size = (int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frameCount = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)
    if outputEdgesFilename is not None:
        outputVideo = cv2.VideoWriter(outputEdgesFilename, cv2.VideoWriter_fourcc(*'DIVX'), fps, size, False) # MPEG-4编码
    staticBW = None
    videoInput.set(cv2.CAP_PROP_POS_FRAMES, int(frameCount*startFrameRate))  # 从视频的3/4处开始读取
    while videoInput.isOpened():
        ret, frame = videoInput.read()
        if ret is False:
            break
        grayFrame = BGR2Gray(frame)
        edgeFrame = Gray2Edge(grayFrame)  # 边缘识别
        if staticBW is None:
            staticBW = edgeFrame  # 初始化staticBW
        else:
            staticBW &= edgeFrame  # 做与运算，不同点会被去掉
        if outputEdgesFilename is not None:
            outputVideo.write(edgeFrame)  # 写入边缘识别结果
        #cv2.imshow('frame', edgeFrame)
        #if cv2.waitKey(2) & 0xFF == ord('q'):
        #   break
    videoInput.release()
    if outputEdgesFilename is not None:
        outputVideo.release()
    return staticBW


videoFilename = '开关柜.mp4'
staticEdges = GetStaticFrame_Edges(videoFilename, startFrameRate=4/5, outputEdgesFilename='output.mp4')  # 获得不动的物体
cv2.imshow('static edges', staticEdges)
cv2.waitKey(2000)
lines = GetLines(staticEdges, 100)
# 获得第一帧
videoInput = cv2.VideoCapture(videoFilename)
_, firstFrame = videoInput.read()
# 向这帧画线
WriteLinesOnImage(firstFrame, lines, 2)
cv2.imshow('result', firstFrame)
cv2.waitKey(0)
cv2.destroyAllWindows()
