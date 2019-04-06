import cv2

def otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged


video = cv2.VideoCapture('开关柜.mp4')
# 获得码率及尺寸
fps = video.get(cv2.CAP_PROP_FPS)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output = cv2.VideoWriter('Output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size, False) # MPEG-4编码

while video.isOpened():
    ret, frame = video.read()
    if ret is False:
        break
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edgeFrame = otsu_canny(grayImg)
    output.write(edgeFrame)
    cv2.imshow('frame', edgeFrame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

video.release()
output.release()
cv2.destroyAllWindows()



'''
img = cv2.imread('img.jpg')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image', grayImg)
cv2.waitKey(1000)
cv2.destroyAllWindows()

edges = otsu_canny(grayImg, lowrate=0.4)
cv2.imwrite('edges.jpg', edges)
cv2.imshow('edges', edges)
cv2.waitKey(2000)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
'''