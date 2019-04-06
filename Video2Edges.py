import cv2

def otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged


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
