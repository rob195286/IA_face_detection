from face_detection import PlaceEmoji
from images_management import resize_image
import cv2 as cv
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--picture_path', dest='path', default='images\\Perso\\g1.jpg', help='Chemin de la photo à analyser')
parser.add_argument('-v', '--is_video', dest='is_video', default=False, help='Indique s\'il faut faire l\'analyse d\'une vidéo ou d\'une image')
args = parser.parse_args()


if(args.is_video):
    pe = PlaceEmoji()
    pe.Play_video(args.path)

else:
    test_image = args.path
    pe = PlaceEmoji()
    result = pe.Get_image_with_faces(test_image)

    if (result.shape[0] > 900 or result.shape[1] > 1500):
        result = resize_image(result, (1280, 760))

    cv.imshow("Faces found", result)
    cv.waitKey(0)
