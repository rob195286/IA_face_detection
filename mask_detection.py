import cv2 as cv
from images_management import direcotry_path


model = cv.CascadeClassifier('Model/mouth.xml')

def check_if_face_is_masked(img):
    result = len(model.detectMultiScale(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 1.5, 5)) == 0
    print(result)
    return result


if __name__ == "__main__" :
    img_name = 'fm2.jpg'
    test_image = direcotry_path + 'Perso\\' + img_name
    result = check_if_face_is_masked(cv.imread(test_image))
    cv.imshow("Faces found", cv.imread(test_image))
    cv.waitKey(0)