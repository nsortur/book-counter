# Count books in an image using OpenCV and TensorFlow

This project uses TensorFlow Hub's CenterNet Object and Keypoints detection model trained on the COCO 2017 dataset, 
paired with OpenCV's Probablistic Hough Transform to count the number of books in an image.

## Running

Install the following packages, then run `python3 count_books.py`
* numpy
* tensorflow
* tensorflow_hub (`pip install --upgrade tensorflow-hub`)
* cv2 (`pip install opencv-python`)

Make sure you're also connected to the internet so TensorFlow Hub can load the model

## Sample results
Horizontal             |  Vertical
:-------------------------:|:-------------------------:
![Stack](./results/stack.png)  |  ![Vertical](./results/vert.png)

![Jumble](./results/jumble.png)

## Previous model
![Vid](./results/book_det_gif.gif)
![Hough](./results/book_det_img.png)