import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2

# todo: optimized for 16:9 or landscape images
# todo: fix resizing (see how to check size)

while True:
    try:
        path = input("Please enter image path from current directory (ex ./assets/IMG_8393.JPG): ")
        f = open(path, 'r')
        f.close()
        break
    except IOError:
        print("Couldn't file file")
        continue

tf.get_logger().setLevel('ERROR')

print('Must use TensorFlow v2.2.0 or higher')
print(f'Using TensorFlow v{tf.__version__}')

with open('./assets/instances_val2017.json') as f:
    dic = json.load(f)
cats = dic['categories']

# tolerance (pixels) between lines
# higher tolerance is less divisions, less books counted
tolerance_hough = 8

img = cv2.imread(path)
# img = cv2.resize(img, (800, 450))
img_tensor = np.expand_dims(img, axis=0)

im_draw = img.copy()
kernel = np.ones((3, 3), np.uint8)

img_blur = cv2.GaussianBlur(img, (15, 15), 0)
edges = cv2.Canny(img_blur, 30, 40, apertureSize=3)
img_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=100, maxLineGap=5)

# initialize vars for hough transform
inits_so_far = []
all_lines = []
hough_count = 0


# checks if a line is close enough to other lines to be considered the same line
def check_same(new_y, pnts_so_far) -> bool:
    same = False
    for init in pnts_so_far:
        same = abs(new_y - init[1]) <= tolerance_hough
        # todo: check horizontal stacking # or abs(x - init[0]) <= tolerance
        if same:
            break
    return same


for line in img_lines:
    x1, y1, x2, y2 = line[0]

    midpoint_y = (y1 + y2) / 2
    # angle of hough line
    theta = np.arctan2(y1 - y2, x1 - x2)

    same_line = check_same(midpoint_y, inits_so_far)
    inits_so_far.append([x1, midpoint_y])

    # makes sure hough lines aren't too close and horizontal enough
    if not same_line and (3.12 < theta < 3.15 or -3.12 < theta < -3.15):
        all_lines.append([(x1, y1), (x2, y2)])
        hough_count += 1
        _ = cv2.line(im_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

# account for top and bottom line double counting
hough_count = hough_count - 1

cv2.putText(im_draw, f'Hough ct: {hough_count}', (img.shape[1] - 400, img.shape[0] - 30),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

print('Loading model...')
detector = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v2_512x512_kpts/1")
print('Model loaded')


# draws bounding rectangle and labels image for book detector
def bound_label(width, height, det_box, det_idx) -> None:
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
def bx_line_intersec(top_left: tuple, bot_right: tuple) -> bool:
    # box segments, going counterclockwise from top left
    seg_left = (top_left, (top_left[0], bot_right[1]))
    seg_bot = ((top_left[0], bot_right[1]), bot_right)
    seg_right = (bot_right, (bot_right[0], top_left[1]))
    seg_top = ((bot_right[0], top_left[1]), top_left)
    box_segments = [seg_left, seg_bot, seg_right, seg_top]

    for line in all_lines:
        for seg in box_segments:
            # if it passes through or if it's fully contained
            intersections = pass_through(line[0], line[1], seg[0], seg[1]) or \
                            (top_left[0] < line[0][0] < bot_right[0] and top_left[1] < line[0][1] < bot_right[1] and
                             top_left[0] < line[1][0] < bot_right[0] and top_left[1] < line[1][1] < bot_right[1])
            if intersections:
                return intersections


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

# initialize vars for cn model
cn_count = 0
box_pts_so_far = []


print('Analyzing image...')
result = detector(img_tensor)
print('Image analyzed')

width = im_draw.shape[1]
height = im_draw.shape[0]

# only classify confident images
res_scores = result['detection_scores'].numpy()[0]
res_scores_confident = np.extract(res_scores >= thresh, res_scores)

boxes = result['detection_boxes'].numpy()

for det_idx in range(len(res_scores_confident)):
    box = boxes[:, det_idx, :]
    # get box bounds from model
    top_left = (int(box[0][1] * width), int(box[0][0] * height))
    bot_right = (int(box[0][3] * width), int(box[0][2] * height))

    # ensure box isn't double counted with a hough line or too close to another box
    clean_box = not bx_line_intersec(top_left, bot_right)
    if get_lbl(det_idx) == "book" and good_box(box) and clean_box:
        bound_label(width, height, box, det_idx)
        cn_count += 1

cv2.putText(im_draw, f'Tf ct: {cn_count}', (im_draw.shape[1] - 140, im_draw.shape[0] - 30),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

# ensure hough count isn't -1
book_ct = hough_count + cn_count if hough_count > 0 else cn_count

cv2.putText(im_draw, f'Final: {book_ct}', (im_draw.shape[1] - 260, im_draw.shape[0] - 60),
            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Hough and CenterNet", im_draw)
print('See popup window and press any key to exit')
cv2.waitKey(0)
