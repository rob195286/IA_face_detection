from enum import Enum
import os
import cv2 as cv

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


path_images_with_mood = direcotry_path + "Images_with_mood\\train\\" + str(Mood.angry) + "\\"
emoji_path = direcotry_path + "Emoji\\"


def get_emoji(mood: Mood):
    return cv.imread(emoji_path + "\\" + str(mood) + ".PNG")


def resize_image(img, new_coord: tuple):
    return cv.resize(img, new_coord)