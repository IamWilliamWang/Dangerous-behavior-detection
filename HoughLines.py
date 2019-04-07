import cv2
import numpy as np

BW = cv2.imread('image_raw.jpg')
BWP = BW.copy()
# staticBW = cv2.imread('staticGrayBW.png', 0)  # 读取单通道灰度图
gray = cv2.cvtColor(BW, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(BW, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('houghline.jpg', BW)



'''
@Deprecated Code
minLineLength = 1000
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for i in range(100):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(BWP,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imwrite('houghlinessP.jpg',BWP)
cv2.destroyAllWindows()
'''