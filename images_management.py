import cv2
from enum import Enum
import matplotlib.pyplot as plt
import os
import numpy as np

direcotry_path = os.getcwd() + "\images\\"

class Mood(Enum):
    angry = "angry"
    disgust = "disgust"
    fear = "fear"
    happy ="happy"
    neutral = "neutral"
    sad = "sad"
    surprise = "surprise"

    def __str__(self):
        return '{0}'.format(self.value)


picture_name = "Training_3908.jpg"
file = direcotry_path + "images_with_mood_face\\train\\" + str(Mood.angry) + "\\" + picture_name
img = cv2.imread(file)
blank = np.zeros(shape=(512,512,3),dtype=np.int16)

def get_all_mood_pictures(mood : Mood):
    return list(os.walk(direcotry_path + str(mood) + "\\"))[0][2]

def put_rectangle_to_image(img):
    return cv2.rectangle(
        img,
         (384,0),
         (510,128),
         (0,0,255),
         5
    )

def put_text_to_image(img, text : str):
    return cv2.putText(
        img,
        text,
        (10,500),
        cv2.FONT_HERSHEY_SIMPLEX,
        4,
        (255,255,255),
        2,
        cv2.LINE_AA
    )


if __name__ == "__main__" :
    #get_all_mood_pictures(Mood.angry)
    plt.imshow(put_text_to_image(blank, 'blablabla'))
    plt.show()