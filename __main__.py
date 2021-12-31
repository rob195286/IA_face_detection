from face_detection import PlaceEmoji
from images_management import direcotry_path, resize_image
import cv2 as cv



img_name = 'g2.jpg'
# picture_name = "Training_3908.jpg"
test_image = direcotry_path + 'Perso\\' + img_name

pe = PlaceEmoji(test_image)
result = pe.get_image_with_faces()

if (result.shape[0] > 900 or result.shape[1] > 1500):
    result = resize_image(result, (1280, 760))
cv.imshow("Faces found", result)
cv.waitKey(0)
