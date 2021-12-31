from fer import FER
from images_management import direcotry_path
import cv2 as cv

model = FER(mtcnn=True)

def get_top_mood(img):
    r, _ = model.top_emotion(img)
    return r

def get_moods(img):
    result = FER().detect_emotions(img)
    result = result[0] if(len(result) > 0) else 0
    return result['emotions']


if __name__ == "__main__" :
    path =  cv.imread(direcotry_path + 'Perso\\bebe.png')
    print(get_moods(path))
