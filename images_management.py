import cv2
from enum import Enum
import matplotlib.pyplot as plt
import os

direcotry_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "\images\\train\\"

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


def get_all_mood_pictures(mood : Mood):
    return list(os.walk(direcotry_path + str(mood) + "\\"))[0][2]



if __name__ == "__main__" :
    get_all_mood_pictures(Mood.angry)
    #plt.imshow(img_array)
    #plt.show()