from images_management import directory_path
import cv2 as cv


def get_top_mood(img, model):
    """
    Récupère l'émotion la plus probable détectée à partir d'une image d'un visage.
    :param img: Image contenant un visage.
    :param model: Model qui se cahrge de faire la détection.
    :return: Renvois
    """
    r, _ = model.top_emotion(img)
    return r


if __name__ == "__main__" :
    path =  cv.imread(directory_path + 'Perso\\bebe.png')

