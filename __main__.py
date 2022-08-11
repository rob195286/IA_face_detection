import argparse

import cv2 as cv

from face_detection import PlaceEmoji
from images_management import resize_image

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--picture_path', dest='path', default='images\\Perso\\g1.jpg',
                    help='Chemin de la photo à analyser')
parser.add_argument('-v', '--is_video', dest='is_video', default=False,
                    help='Indique s\'il faut faire l\'analyse d\'une vidéo ou d\'une image')
args = parser.parse_args()

# ---------------------------------------------- Avec camera
if args.is_video:
    pe = PlaceEmoji()
    pe.play_video()
# ---------------------------------------------- Sans camera
else:
    test_image = args.path
    pe = PlaceEmoji()
    result = pe.get_image_with_faces(test_image)

    if result.shape[0] > 900 or result.shape[1] > 1500:  # Redimensionne image trop grande
        result = resize_image(result, (1280, 760))

    cv.imshow("Faces found", result)
    cv.waitKey(0)
