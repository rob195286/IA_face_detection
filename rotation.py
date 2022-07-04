import cv2 as cv
import numpy as np
from math import sin, cos, radians, degrees


im = cv.imread('images\\Perso\\r1.jpg')

model = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')


def Rotation2(img):
    def computeDistance(origin, destination):
        """Find the distance between two points in the Cartesian coordinates"""
        distance = origin - destination
        distance = np.sqrt(np.sum(np.multiply(distance, distance)))
        return distance

    detected_face_toGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eyes = model.detectMultiScale(detected_face_toGray, 1.1, 10)

    # If less than 2 eyes detected -> None
    if len(eyes) >= 2:
        # contient les coord x et y du début du rectangle.
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        # Detect which eye is Left or Right based on position x
        if eye_1[0] < eye_2[0]:
            L_eye = eye_1
            R_eye = eye_2
        else:
            L_eye = eye_2
            R_eye = eye_1

        # Find midpoint of eyes as (x + w/2) and (y + h/2), on prend le point en x (horizontal) on l'étend de w, pareil avec y (vertical).
        L_eye = (int(L_eye[0] + (L_eye[2] / 2)), int(L_eye[1] + (L_eye[3] / 2)))
        R_eye = (int(R_eye[0] + (R_eye[2] / 2)), int(R_eye[1] + (R_eye[3] / 2)))

        # Find rotation direction, via le centre des yeux.
        L_eye_x, L_eye_y = L_eye
        R_eye_x, R_eye_y = R_eye
        print('L_eye_x :', L_eye_x)
        print('L_eye_y :', L_eye_y)
        print(' R_eye_x, : ',  R_eye_x)
        print(' R_eye_y, : ',  R_eye_y)

        if L_eye_y > R_eye_y:
            triangle_point = (R_eye_x, L_eye_y)
            direction = -1  # Rotate clockwise (antitrigo)
        else:
            triangle_point = (L_eye_x, R_eye_y)
            direction = 1  # Rotate counter-clockwise

        a = computeDistance(np.array(L_eye), np.array(triangle_point))
        b = computeDistance(np.array(R_eye), np.array(triangle_point))
        c = computeDistance(np.array(R_eye), np.array(L_eye))  # Hypotenuse

        # If head is not horizontal, apply cosine rule to compute angle
        if b != 0 and a != 0:
            cos_theta = (b * b + c * c - a * a) / (2 * b * c)
            angle = degrees(np.arccos(cos_theta))
            if direction == -1:
                angle = 90 - angle
            return direction * angle
        return None
    return None


def Rotation(img):
    """
    for (ex, ey, ew, eh) in model.detectMultiScale(img):
        cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        """
    (ex, ey, ew, eh) = model.detectMultiScale(img)[2]
    cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    print( model.detectMultiScale(img))
    cv.imshow("Faces found", img)
    cv.waitKey(0)

def Rotation3(img):
    detected_face_toGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    eyes = model.detectMultiScale(detected_face_toGray, 1.1, 10)

    # If less than 2 eyes detected -> None
    if len(eyes) >= 2:
        # contient les coord x et y du début du rectangle.
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        # Detect which eye is Left or Right based on position x
        if eye_1[0] < eye_2[0]:
            L_eye = eye_1
            R_eye = eye_2
        else:
            L_eye = eye_2
            R_eye = eye_1

        # Find midpoint of eyes as (x + w/2) and (y + h/2), on prend le point en x (horizontal) on l'étend de w, pareil avec y (vertical).
        L_eye = (int(L_eye[0] + (L_eye[2] / 2)), int(L_eye[1] + (L_eye[3] / 2)))
        R_eye = (int(R_eye[0] + (R_eye[2] / 2)), int(R_eye[1] + (R_eye[3] / 2)))

        # Find rotation direction, via le centre des yeux.
        L_eye_x, L_eye_y = L_eye
        R_eye_x, R_eye_y = R_eye

        cv.circle(img, (R_eye_x, R_eye_y), radius=5, color=(0, 255, 0))
        print(model.detectMultiScale(img))
        cv.imshow("Faces found", img)
        cv.waitKey(0)


        if L_eye_y > R_eye_y:
            triangle_point = (R_eye_x, L_eye_y)
            direction = -1  # Rotate clockwise (antitrigo)
        else:
            triangle_point = (L_eye_x, R_eye_y)
            direction = 1  # Rotate counter-clockwise

        print('L_eye_x :', L_eye_x)
        print('L_eye_y :', L_eye_y)
        print(' R_eye_x, : ', R_eye_x)
        print(' R_eye_y, : ', R_eye_y)



if __name__ == "__main__" :
    #print(Rotation2(im))
    #Rotation(im)
    Rotation3(im)