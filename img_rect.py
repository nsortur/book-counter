import cv2
import numpy as np

path = './assets/IMG_8389.JPG'
img = cv2.imread(path)
img = cv2.resize(img, (600, 800))
im_draw = img.copy()

kernel = np.ones((3, 3), np.uint8)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
edges = cv2.Canny(img_blur, 50, 50)

def draw_contours(canny):
    # find contours
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            print(area)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.005 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            asp_ratio = w / float(h)
            if asp_ratio > 2 or asp_ratio < 0.25:
                cv2.rectangle(im_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #cv2.putText(im_draw, str(area), (x+w//2, y+h//2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)


draw_contours(edges)
stack = np.hstack([np.stack([img_blur, img_blur, img_blur], axis=2),
                   np.stack([edges, edges, edges], axis=2),
                   im_draw])
cv2.imshow("Books", stack)

cv2.waitKey(0)
