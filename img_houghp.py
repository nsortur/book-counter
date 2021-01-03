import cv2
import numpy as np

path = './assets/IMG_8393.JPG'
img = cv2.imread(path)
img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (15, 15), 0)
edges = cv2.Canny(img_blur, 30, 40, apertureSize=3)
im_draw = img.copy()

inits_so_far = []
book_count = 0
tolerance_hough = 8


# checks if a line is close enough to other lines to be considered the same line
def check_same(new_y, pnts_so_far) -> bool:
    same = False
    for init in pnts_so_far:
        same = abs(new_y - init[1]) <= tolerance_hough
        # todo: check horizontal stacking # or abs(x - init[0]) <= tolerance
        if same:
            break
    return same


lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=100, maxLineGap=5)
for line in lines:
    x1, y1, x2, y2 = line[0]
    midpoint_y = (y1 + y2) / 2
    theta = np.arctan2(y1 - y2, x1 - x2)
    same_line = check_same(midpoint_y, inits_so_far)
    inits_so_far.append([x1, midpoint_y])

    if not same_line and (3.12 < theta < 3.15 or -3.12 < theta < -3.15):
        print(theta)
        cv2.line(im_draw, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(im_draw, str(round(theta, 3)), (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        book_count += 1

print(book_count)
cv2.imshow("Edge", edges)
cv2.imshow("Drawed", im_draw)
cv2.waitKey(0)
