from face_detection import PlaceEmoji
from images_management import resize_image
import cv2 as cv
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--picture_path', dest='path', help='Chemin de la photo à analyser')
args = parser.parse_args()


test_image = args.path

pe = PlaceEmoji(test_image)
result = pe.get_image_with_faces()

if (result.shape[0] > 900 or result.shape[1] > 1500):
    result = resize_image(result, (1280, 760))

cv.imshow("Faces found", result)
cv.waitKey(0)
