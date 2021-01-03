import cv2
import numpy as np

path = './assets/IMG_8393.JPG'
img = cv2.imread(path)
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
edges = cv2.Canny(img_blur, 40, 120, apertureSize=3)
im_draw = img.copy()

line_ct = 0
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 170, minLineLength=100, maxLineGap=5)
for line in lines:
    line_ct += 1
    x1, y1, x2, y2 = line[0]
    cv2.line(im_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

print(line_ct)
cv2.imshow("Edge", edges)
cv2.imshow("Drawed", im_draw)
cv2.waitKey(0)
