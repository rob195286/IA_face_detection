import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from images_management import path_images_with_mood, direcotry_path, Mood


img = cv2.imread(path_images_with_mood)
blank = np.zeros(shape=(512,512,3),dtype=np.int16)

def get_all_mood_pictures(mood : Mood):
    return list(os.walk(direcotry_path + str(mood) + "\\"))[0][2]

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