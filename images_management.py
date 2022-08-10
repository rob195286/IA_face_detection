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

emoji_img = {
    str(Mood.angry) : cv.imread(emoji_path + "\\" + str(Mood.angry) + ".PNG"),
    str(Mood.disgust) : cv.imread(emoji_path + "\\" + str(Mood.disgust) + ".PNG"),
    str(Mood.fear) : cv.imread(emoji_path + "\\" + str(Mood.fear) + ".PNG"),
    str(Mood.happy) : cv.imread(emoji_path + "\\" + str(Mood.happy) + ".PNG"),
    str(Mood.neutral) : cv.imread(emoji_path + "\\" + str(Mood.neutral) + ".PNG"),
    str(Mood.sad) : cv.imread(emoji_path + "\\" + str(Mood.sad) + ".PNG"),
    str(Mood.surprise) : cv.imread(emoji_path + "\\" + str(Mood.surprise) + ".PNG"),
    'Mask' : cv.imread(emoji_path + "\\" + "Mask.PNG")
}


def get_emoji(mood: str, new_size: tuple):
    """
    Récupère à partir du dictionnaire "emoji_img" l'image de l'émoji correspondant.
    :param mood: String de l'émotion dont on veut en récupérer l'émoji (7 émotions + le masque).
    :param new_coord: Nouvelle dimension que l'émoji doit avoir pour correspondre à la tailel du visage.
    :return: L'émoji correspondante, redimensionnée.
    """
    return resize_image(emoji_img[mood], new_size)

def resize_image(img, new_coord: tuple):
    return cv.resize(img, new_coord)

if __name__ == "__main__" :
    file_name = emoji_path + "\\sad.PNG"

    src = cv.imread(file_name, 1)
    tmp = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # Applying thresholding technique
    _, alpha = cv.threshold(tmp, 10, 255, cv.THRESH_BINARY)
    # Using cv2.split() to split channels
    # of coloured image
    b, g, r = cv.split(src)
    # Making list of Red, Green, Blue
    # Channels and alpha
    rgba = [b, g, r, alpha]
    # Using cv2.merge() to merge rgba
    # into a coloured/multi-channeled image
    dst = cv.merge(rgba, 4)
    cv.imshow('img', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()