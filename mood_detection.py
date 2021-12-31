from fer import FER
from images_management import direcotry_path
import cv2 as cv

def get_mood(img):
    result = FER().detect_emotions(img)
    result = result[0] if(len(result) > 0) else 0
    return max(result['emotions'], key=result['emotions'].get) if(result != 0) else 0

def get_moods(img):
    result = FER().detect_emotions(img)
    result = result[0] if(len(result) > 0) else 0
    return result['emotions']


if __name__ == "__main__" :
    path =  cv.imread(direcotry_path + 'perso\\bebe.png')
    print(get_moods(path))
