import sys
import random
import numpy
import cv2

# set sample image name
image_name = 'face1'
if len(sys.argv) > 1:
    image_name = sys.argv[1]

# xmas hats
hats = []
for i in range(4):
    hats.append(cv2.imread('images/hat%d.png' % i, -1))

# opencv face detect
sample_image = cv2.imread('images/' + image_name + '.jpg')

def face_detect(image, classifier):
    pattern = cv2.CascadeClassifier(classifier)
    return pattern.detectMultiScale(image,
                                    scaleFactor=1.1,
                                    minNeighbors=8,
                                    minSize=(50, 50))

# while using multiple classifiers we need these utility functions
# to merge results and remove duplicates
def is_duplicate(face1, face2):
    a = numpy.array((face1[0], face1[1]))
    b = numpy.array((face2[0], face2[1]))
    distance = numpy.linalg.norm(a - b)
    return (distance < 60) # the magic threshold

def merge_faces(faces1, faces2):
    for face2 in faces2:
        is_new_face = True
        for face1 in faces1:
            if is_duplicate(face1, face2):
                is_new_face = False
                break
        if is_new_face:
            faces1 = numpy.append(faces1, [face2], axis = 0)
    return faces1

faces1 = face_detect(sample_image, 'classifiers/haarcascades/haarcascade_frontalface_alt.xml')
faces2 = face_detect(sample_image, 'classifiers/haarcascades/haarcascade_profileface.xml')
faces = merge_faces(faces1, faces2)

# put hats on
for face in faces:
    # choose random hat
    hat = random.choice(hats)
    # adjust the hat size
    scale = face[3] / hat.shape[0] * 1.25
    hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale)
    # calculate hat position via face detection, avoid edge cut
    x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2)
    y_offset = int(face[1] - hat.shape[0] / 2)
    x1, x2 = max(x_offset, 0), min(x_offset + hat.shape[1], sample_image.shape[1])
    y1, y2 = max(y_offset, 0), min(y_offset + hat.shape[0], sample_image.shape[0])
    hat_x1 = max(0, -x_offset)
    hat_x2 = hat_x1 + x2 - x1
    hat_y1 = max(0, -y_offset)
    hat_y2 = hat_y1 + y2 - y1
    # transparency
    alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
    alpha = 1 - alpha_h
    # merge texture
    for c in range(0, 3):
        sample_image[y1:y2, x1:x2, c] = (alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] + alpha * sample_image[y1:y2, x1:x2, c])

# write output image
cv2.imwrite('output/' + image_name + '-xmas.png', sample_image)
