import copy
import cv2 as cv
from images_management import direcotry_path, emoji_path, Mood
from mood_detection import get_top_mood



class PlaceEmoji():
    def __init__(self, img, draw_rectangle=False, classifier = None):
        self.__initial_img = cv.imread(img)
        self.__emojited_img = copy.deepcopy(self.__initial_img)
        if(classifier is None):
            self.classifier = cv.CascadeClassifier('Model/haarcascade_frontalface_default.xml')
        else:
            self.classifier = classifier
        self.rectangles_coordinates = []
        self.__faces_data = []
        self.__draw_rectangle(draw_rectangle)


    def __draw_rectangle(self, draw_rectangle: bool):
        rectangle_detected = self.classifier.detectMultiScale(
            self.__initial_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in rectangle_detected:
            if (draw_rectangle):
                cv.rectangle(self.__initial_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.rectangles_coordinates.append({
                'rectangle_start_coord_x' : x,
                'rectangle_start_coord_y' : y,
                'rectangle_end_coord_x' : x + w,
                'rectangle_end_coord_y' : y + h
            })

    def __change_to_gray_scale(self):
        return cv.cvtColor(self.__initial_img, cv.COLOR_BGR2GRAY)

    def __get_emoji(self, mood: Mood):
        return cv.imread(emoji_path + "\\" + str(mood) + ".PNG")

    def __resize_image(self, img, new_coord: dict):
        width = new_coord['rectangle_end_coord_x'] - new_coord['rectangle_start_coord_x']
        height = new_coord['rectangle_end_coord_y'] - new_coord['rectangle_start_coord_y']
        return cv.resize(img, (width, height))

    def __place_emoji(self):
        for face_data in self.__faces_data:
            rectangle_coord = face_data['face_coord']
            self.__emojited_img[
                       face_data['face_coord']['rectangle_start_coord_y']: face_data['face_coord']['rectangle_end_coord_y'],
                       face_data['face_coord']['rectangle_start_coord_x']: face_data['face_coord']['rectangle_end_coord_x']
                       ] \
                = self.__resize_image(self.__get_emoji(face_data['mood']), rectangle_coord)

    def __select_face_on_image(self):
        self.__faces_data = []
        image_copy = self.__initial_img.copy()
        for rectangle_of_face in self.rectangles_coordinates:
            face = image_copy[
                   rectangle_of_face['rectangle_start_coord_y']: rectangle_of_face['rectangle_end_coord_y'],
                   rectangle_of_face['rectangle_start_coord_x']: rectangle_of_face['rectangle_end_coord_x']
                   ]
            mood = get_top_mood(face)
            if (mood == None):
                print('pas de mood trouvé')
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
        self.__select_face_on_image()
        self.__place_emoji()
        return self.__emojited_img



if __name__ == "__main__" :
    img_name = 'g4.jpg'
    #picture_name = "Training_3908.jpg"
    test_image = direcotry_path + '\\Perso\\' + img_name

    pe = PlaceEmoji(test_image)

    cv.imshow("Faces found", pe.get_image_with_faces())
    cv.waitKey(0)

    for f in pe.get_face_data():
        print(f['mood'])
        cv.imshow("Faces found", f['face'])
        cv.waitKey(0)
