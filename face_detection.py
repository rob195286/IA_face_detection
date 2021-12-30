import cv2 as cv
from images_management import direcotry_path


img_start_coord = 0
img_end_coord = 0

class PlaceEmoji():
    def __init__(self, img, classifier = None):
        self.img = cv.imread(img)
        if(not classifier):
            self.classifier = self.__face_classifier()
        else:
            self.classifier = classifier
        self.rectangles_coordinates = []
        self.faces = []


    def __face_classifier(self):
        cascade_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        return cascade_classifier.detectMultiScale(
            self.img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

    def __draw_rectangle(self):
        for (x, y, w, h) in self.classifier:
            cv.rectangle(self.img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.rectangles_coordinates.append({
                'rectangle_start_coord' : (x,y),
                'rectangle_end_coord' : (x + w, y + h),
            })

    def __change_to_gray_scale(self):
        return cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

    def __select_face(self):
        image_copy = self.img.copy()
        for rectangle_of_face in self.rectangles_coordinates:
            self.faces.append({
                'mood': 0,
                'face': image_copy[rectangle_of_face['rectangle_start_coord'][1] : rectangle_of_face['rectangle_end_coord'][1],
                        rectangle_of_face['rectangle_start_coord'][0] : rectangle_of_face['rectangle_end_coord'][0]]
            })

    def get_faces(self):
        self.__draw_rectangle()
        self.__select_face()
        for x in self.faces:
            cv.imshow("Faces found", x['face'])
            cv.waitKey(0)
        return self.img



if __name__ == "__main__" :
    test_image = direcotry_path + '\\perso\\g2.jpg'
    #test_image = cv.imread(test_image)
    #test_image = change_to_gray_scale(test_image)
    pe = PlaceEmoji(test_image)
    #draw_rectangle(test_image)
    pe.get_faces()
    #cv.imshow("Faces found", pe.get_faces())
   # cv.waitKey(0)
