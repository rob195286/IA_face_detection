from math import atan, degrees

import cv2 as cv


model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

def find_rotation_angle(img):
    tmp = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eyes = model.detectMultiScale(tmp)
    if len(eyes) >= 2:
        # contient les coord x et y du début du rectangle.
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        # Détecte lequel est l'oeil de droite et de gauche.
        if eye_1[0] < eye_2[0]:
            L_eye = eye_1
            R_eye = eye_2
        else:
            L_eye = eye_2
            R_eye = eye_1
        (x, y, w, h) = L_eye
        (x2, y2, w2, h2) = R_eye
        # x -> horizontal et y vertical. Changé y fait descendre monté et en x va de gauche à droite.
        left_eye_center = (x + w // 2, y + h // 2)
        right_eye_center = (x2 + w2 // 2, y2 + h2 // 2)
        #                   Coordonnées en y gauche - y droite             /   x gauche - x droite.
        rotation = degrees(atan((left_eye_center[1] - right_eye_center[1]) / (left_eye_center[0] - right_eye_center[0])))
        # Formule = (yg - yd)/(xg - xd)
        return -int(rotation)
    return 0

def rotate_image(img, angle : int):
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D(center=(w / 2, h / 2), angle=angle, scale=1.0) # Récupère la transformation affine.
    return cv.warpAffine(img, M, (w, h)) # Applique cette transformation matricielle sur l'image.



if __name__ == "__main__":
    test_image = 'images\\Perso\\r1.jpg'
   # test_image = directory_path + 'Mask\\without_mask\\' + '0_0_aidai_0136.jpg'
    im = cv.imread(test_image)

    image_rotated = rotate_image(im, find_rotation_angle(im))

    cv.imshow("Faces found", image_rotated)
    cv.waitKey(0)
