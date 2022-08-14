import os

import cv2 as cv

directory_path = os.getcwd() + "\images\\"
emoji_path = directory_path + "Emoji\\"

emoji_img = {
    "angry": cv.imread(emoji_path + "\\angry.PNG"),
    "disgust": cv.imread(emoji_path + "\\disgust.PNG"),
    "fear": cv.imread(emoji_path + "\\fear.PNG"),
    "happy": cv.imread(emoji_path + "\\happy.PNG"),
    "neutral": cv.imread(emoji_path + "\\neutral.PNG"),
    "sad": cv.imread(emoji_path + "\\sad.PNG"),
    "surprise": cv.imread(emoji_path + "\\surprise.PNG"),
    'Mask': cv.imread(emoji_path + "\\" + "Mask.PNG")
}


def get_emoji(mood: str, new_size: tuple):
    """
    Récupère à partir du dictionnaire "emoji_img" l'image de l'émoji correspondant.
    :param mood: String de l'émotion dont on veut en récupérer l'émoji (7 émotions + le masque).
    :param new_size: Nouvelle dimension que l'émoji doit avoir pour correspondre à la tailel du visage.
    :return: L'émoji correspondante, redimensionnée.
    """
    return resize_image(emoji_img[mood], new_size)


def resize_image(img, new_coord: tuple):
    return cv.resize(img, new_coord)


if __name__ == "__main__":
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
