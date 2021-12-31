from fer import FER
from images_management import direcotry_path
import cv2 as cv

def get_mood(img):
    r, _ = FER(mtcnn=True).top_emotion(img)
    return r

def get_moods(img):
    result = FER().detect_emotions(img)
    result = result[0] if(len(result) > 0) else 0
    return result['emotions']


if __name__ == "__main__" :
    path =  cv.imread(direcotry_path + 'perso\\bebe.png')
    print(get_moods(path))
