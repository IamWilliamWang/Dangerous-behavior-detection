import cv2
import numpy as np

# 自动canny函数
def Otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged

def BGR2Gray(BGR):
    return cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)

def Gray2Edge(grayFrame):
    return Otsu_canny(grayFrame)


# 打开输入输出视频文件
videoInput = cv2.VideoCapture('开关柜.mp4')
# 获得码率及尺寸
fps = videoInput.get(cv2.CAP_PROP_FPS)
size = (int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#output = cv2.VideoWriter('Output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size, False) # MPEG-4编码

frameCount = videoInput.get(cv2.CAP_PROP_FRAME_COUNT)
staticBW = None
videoInput.set(cv2.CAP_PROP_POS_FRAMES, int(frameCount*3/4)) #从视频的3/4处开始截取
while videoInput.isOpened():
    ret, frame = videoInput.read()
    if ret is False:
        break
    grayImg = BGR2Gray(frame)  # 变灰度图
    edgeFrame = Gray2Edge(grayImg)  # 边缘识别
    if staticBW is None:
        staticBW = edgeFrame
    else:
        staticBW &= edgeFrame
    #output.write(edgeFrame)  # 写入边缘识别结果
    #cv2.imshow('frame', edgeFrame)
    #if cv2.waitKey(2) & 0xFF == ord('q'):
    #   break

videoInput.release()
#output.release()
cv2.imshow('static BW', staticBW)
cv2.waitKey(0)
cv2.destroyAllWindows()

