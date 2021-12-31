import cv2 as cv
from images_management import direcotry_path
from mood_detection import get_mood


img_name = 'g3.jpg'
img_start_coord = 0
img_end_coord = 0

class PlaceEmoji():
    def __init__(self, img, classifier = None):
        self.__img = cv.imread(img)
        if(classifier is None):
            self.classifier = cv.CascadeClassifier('Model/haarcascade_frontalface_default.xml')
        else:
            self.classifier = classifier
        self.rectangles_coordinates = []
        self.__faces_data = []
        self.__draw_rectangle()


    def __draw_rectangle(self):
        rectangle_detected = self.classifier.detectMultiScale(
            self.__img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in rectangle_detected:
            cv.rectangle(self.__img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.rectangles_coordinates.append({
                'rectangle_start_coord_x' : x,
                'rectangle_start_coord_y' : y,
                'rectangle_end_coord_x' : x + w,
                'rectangle_end_coord_y' : y + h
            })

    def __change_to_gray_scale(self):
        return cv.cvtColor(self.__img, cv.COLOR_BGR2GRAY)

    def __select_face_on_image(self):
        image_copy = self.__img.copy()
        for rectangle_of_face in self.rectangles_coordinates:
            face = image_copy[
                   rectangle_of_face['rectangle_start_coord_y']: rectangle_of_face['rectangle_end_coord_y'],
                   rectangle_of_face['rectangle_start_coord_x']: rectangle_of_face['rectangle_end_coord_x']
                   ]
            mood = get_mood(face)
            if (mood == 0):
                continue
            self.__faces_data.append({
                'mood': mood,
                'face': face,
                'face_coord': rectangle_of_face
            })

    def get_faces(self):
        self.__select_face_on_image()
        return [f['face'] for f in self.__faces_data]

    def get_face_data(self):
        self.__select_face_on_image()
        return self.__faces_data

    def get_image_with_faces(self):
        return self.__img



if __name__ == "__main__" :
    test_image = direcotry_path + '\\perso\\' + img_name
    #test_image = cv.imread(test_image)
    #test_image = change_to_gray_scale(test_image)
    pe = PlaceEmoji(test_image)

    cv.imshow("Faces found", pe.get_image_with_faces())
    cv.waitKey(0)

    for f in pe.get_face_data():
        print(f['mood'])
        cv.imshow("Faces found", f['face'])
        cv.waitKey(0)
