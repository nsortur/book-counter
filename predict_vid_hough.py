import cv2
import numpy as np

# counts books in a vertical stack

# tolerance (pixels) between lines
# higher tolerance is less divisions, less books counted
tolerance = 7

cap = cv2.VideoCapture(0)
# define width
cap.set(3, 640)
# define height
cap.set(4, 480)
# change brightness
cap.set(10, 5)


while True:
    success, img = cap.read()
    # img = cv2.resize(frame, (600, 800))

    if success:
        im_draw = img.copy()
        kernel = np.ones((3, 3), np.uint8)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img, (15, 15), 0)
        edges = cv2.Canny(img_blur, 10, 10, apertureSize=3)
        img_lines = cv2.HoughLines(edges, 1, np.pi / 180, 170)
        inits_so_far = []
        book_count = 0


        def check_same(new_y, pnts_so_far):
            same = False
            # calculate slope
            for init in pnts_so_far:
                same = abs(new_y - init[1]) <= tolerance
                # todo: check horizontal stacking # or abs(x - init[0]) <= tolerance
                if same:
                    break
            return same

        if np.array(img_lines).size != 1:
            for line in img_lines:
                rho, theta = line[0]
                # convert polar to linear
                a = np.cos(theta)
                b = np.sin(theta)
                # gives origin of image
                x0 = a * rho
                y0 = b * rho

                # get lines, (x1, y1) -> (x2, y2)
                # rcos(theta)-1000 * sin(theta)
                x1 = int(x0 + 1000 * (-b))
                # rsin(theta)+1000 * cos(theta)
                y1 = int(y0 + 1000 * a)

                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)

                # check if midpoints are in tolerance
                midpoint_y = (y1 + y2) / 2
                print('Y location: ', midpoint_y)
                same_line = check_same(midpoint_y, inits_so_far)
                inits_so_far.append([x1, midpoint_y])

                # makes sure hough lines are horizontal enough (pi / 2 with tolerance)
                if not same_line and 1.48 < theta < 1.66:
                    book_count += 1
                    #cv2.line(im_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(im_draw, str(y1), (20, y1),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

        # subtract 1 to account for top and bottom of stack
        cv2.putText(im_draw, f'Books: {book_count - 1}', (img.shape[1] - 140, img.shape[0] - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Video", im_draw)

    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break
