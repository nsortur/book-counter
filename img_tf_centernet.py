import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# todo: return types for functions
# todo: only count detections w/o hough line pass throughs
# todo: calculate top left and bot right only once
# todo: houghlinesp (short segments)

tf.get_logger().setLevel('ERROR')
print(f'Using TensorFlow v{tf.__version__}')

with open('./assets/instances_val2017.json') as f:
    dic = json.load(f)
cats = dic['categories']

# tolerance (pixels) between lines
# higher tolerance is less divisions, less books counted
tolerance_hough = 8

path = './assets/books_hor.JPG'
img = cv2.imread(path)
# img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
img_tensor = np.expand_dims(img, axis=0)

im_draw = img.copy()
kernel = np.ones((3, 3), np.uint8)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img, (15, 15), 0)
edges = cv2.Canny(img_blur, 40, 40, apertureSize=3)
img_lines = cv2.HoughLines(edges, 1, np.pi / 180, 170)

inits_so_far = []
all_lines = []
hough_count = 0


def check_same(new_y, pnts_so_far):
    same = False
    for init in pnts_so_far:
        same = abs(new_y - init[1]) <= tolerance_hough
        # todo: check horizontal stacking # or abs(x - init[0]) <= tolerance
        if same:
            break
    return same


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

    same_line = check_same(midpoint_y, inits_so_far)
    inits_so_far.append([x1, midpoint_y])

    all_lines.append([(x1, y1), (x2, y2)])

    # makes sure hough lines are horizontal enough (pi / 2 with tolerance)
    if not same_line and 1.5 < theta < 1.64:
        hough_count += 1
        _ = cv2.line(im_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
        _ = cv2.putText(im_draw, str(y1), (20, y1),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

# subtract 1 to account for top and bottom of stack
hough_count = hough_count - 1
cv2.putText(im_draw, f'Hough ct: {hough_count}', (img.shape[1] - 400, img.shape[0] - 30),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

# images with this many or more hough lines are considered stacks
hough_sig = hough_count > 4

print('Loading model...')
detector = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1")
print('Model loaded')


# draws bounding rectangle and labels image for book detector
def bound_label(width, height, det_box, det_idx):
    x1, y1 = int(det_box[0][1] * width), int(det_box[0][0] * height)
    x2, y2 = int(det_box[0][3] * width), int(det_box[0][2] * height)

    _ = cv2.rectangle(im_draw,
                      (x1, y1),
                      (x2, y2),
                      (0, 255, 0),
                      thickness=2)
    _ = cv2.putText(im_draw,
                    get_lbl(det_idx),
                    (int((x2 + x1) / 2) - 30, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    box_pts_so_far.append([x1, y1])


# gets the text label for a current index from json dict
def get_lbl(curr) -> str:
    res_idx = result['detection_classes'].numpy()[0][curr]
    res = filter(lambda x: x['id'] == res_idx, cats)
    return list(res)[0]['name']


# check if a bounding box is within a certain x, y tolerance to be counted as the same book
def good_box(bx) -> bool:
    good = True
    x1, y1 = int(bx[0][1] * width), int(bx[0][0] * height)
    for pt in box_pts_so_far:
        good = abs(x1 - pt[0]) >= tolerance and abs(y1 - pt[1]) >= tolerance
        if not good:
            break
    return good


# check if any line segments intersect a given box
def bx_line_intersec(top_left, bot_right) -> bool:



# check if line intersects another line
def pass_through(a: tuple, b: tuple, c: tuple, d: tuple) -> bool:
    return pt_help(a, c, d) != pt_help(b, c, d) and pt_help(a, b, c) != pt_help(a, b, d)


# helper for pass through
def pt_help(a: tuple, b: tuple, c: tuple):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


# threshold (only take classifications with this or greater confidence)
thresh = 0.18
# tolerance (pixels) between top left detections
# higher tolerance is less books counted
tolerance = 0

tf_count = 0
box_pts_so_far = []

# only forward propagate tf model if image is not purely a stack
if not hough_sig:
    print('Analyzing image...')
    result = detector(img_tensor)
    print('Image analyzed')

    width = im_draw.shape[1]
    height = im_draw.shape[0]

    # only classify confident images
    res_scores = result['detection_scores'].numpy()[0]
    res_scores_confident = np.extract(res_scores >= thresh, res_scores)
    print('Original scores: ', res_scores)
    print('Confident scores: ', res_scores_confident)

    boxes = result['detection_boxes'].numpy()

    for det_idx in range(len(res_scores_confident)):
        box = boxes[:, det_idx, :]
        top_left = (int(box[0][1] * width), int(box[0][0] * height))
        bot_right = (int(box[0][3] * width), int(box[0][2] * height))

        clean_box = not bx_line_intersec(top_left, bot_right)

        if get_lbl(det_idx) == "book" and good_box(box) and clean_box:
            bound_label(width, height, box, det_idx)
            tf_count += 1

    cv2.putText(im_draw, f'Tf ct: {tf_count}', (im_draw.shape[1] - 140, im_draw.shape[0] - 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

book_ct = hough_count if hough_sig else tf_count

cv2.putText(im_draw, f'Final: {book_ct}', (im_draw.shape[1] - 260, im_draw.shape[0] - 60),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Hough and CenterNet", im_draw)
cv2.waitKey(0)
