import copy
import cv2 as cv
from images_management import get_emoji, resize_image, direcotry_path
from mood_detection import get_top_mood
from mask_detection import check_if_face_is_masked
import face_recognition


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
        # multi scale signifie que l'algo va passer plusieurs dois dans des sous régions
        # afin de détecter des visages de taille variante.
        rectangle_detected = self.classifier.detectMultiScale(
            self.__initial_img,
            scaleFactor=1.1, # % de réduction de l'image
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

    def __change_to_gray_scale(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def __place_emoji(self):
        for face_data in self.__faces_data:
            width = face_data['face_coord']['rectangle_end_coord_x'] - face_data['face_coord']['rectangle_start_coord_x']
            height = face_data['face_coord']['rectangle_end_coord_y'] - face_data['face_coord']['rectangle_start_coord_y']
            self.__emojited_img[
                       face_data['face_coord']['rectangle_start_coord_y']: face_data['face_coord']['rectangle_end_coord_y'],
                       face_data['face_coord']['rectangle_start_coord_x']: face_data['face_coord']['rectangle_end_coord_x']
                       ] \
                = resize_image(get_emoji(face_data['mood']), (width, height))

    def __select_face_on_image(self):
        self.__faces_data = []
        image_copy = self.__initial_img.copy()
        for rectangle_of_face in self.rectangles_coordinates:
            face = image_copy[
                   rectangle_of_face['rectangle_start_coord_y']: rectangle_of_face['rectangle_end_coord_y'],
                   rectangle_of_face['rectangle_start_coord_x']: rectangle_of_face['rectangle_end_coord_x']
                   ]
            if (get_top_mood(face) == None):
                print('pas de mood trouvé')
                continue
            mood = get_top_mood(face)
            #mood = 'mask' if(check_if_face_is_masked(face)) else get_top_mood(face)
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

    def play_video(self, video='test.mp4'):
        cap = cv.VideoCapture(video)
        while True:
            _, frame = cap.read()  # prend une frame, la première valeur indique si elle a été prise, pas important pour nous.
            rgb_frame = frame[:, :, ::-1]  # Convertis en RBG pour pouvoir être utilisé
            face_locations = face_recognition.face_locations(rgb_frame)  # Trouve les visages dans l'image

            for top, right, bottom, left in face_locations:
                cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.imshow('Video', frame)
            if cv.waitKey(25) == 13:  # Wait for Enter key to stop
                break



if __name__ == "__main__" :
    p = PlaceEmoji( direcotry_path + 'Perso\\' + 'gm1.jpg')
    p.play_video()
    '''
    img_name = 'gm1.jpg'
    # picture_name = "Training_3908.jpg"
    test_image = direcotry_path + 'Perso\\' + img_name

    pe = PlaceEmoji(test_image)
    result = pe.get_image_with_faces()

    if (result.shape[0] > 900 or result.shape[1] > 1500):
        result = resize_image(result, (1280, 760))
    cv.imshow("Faces found", result)
    cv.waitKey(0)
'''