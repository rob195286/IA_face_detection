from deepface import DeepFace
import cv2 as cv
from images_management import direcotry_path


path = direcotry_path + "Images_with_mood\\train\\angry\\" + "Training_3908.jpg"
path2 = direcotry_path + "Perso\\bebe.png"
img =  cv.imread(path)
prediction = DeepFace.analyze(img, enforce_detection=False)


if __name__ == "__main__" :
    print(prediction)
"""
    cv.imshow("Faces found", img)
    cv.waitKey(0)

"""