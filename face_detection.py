import copy

import cv2 as cv
import face_recognition
from fer import FER

from images_management import get_emoji, directory_path, resize_image
from mask_detection import face_is_masked
from mood_detection import get_top_mood


class PlaceEmoji:
    def __init__(self, faces_recognition_classifier=None):
        if faces_recognition_classifier is None:
            self.faces_recognition_classifier = cv.CascadeClassifier(
                cv.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Model pour reconnaître plusieurs viages.
        else:
            self.faces_recognition_classifier = faces_recognition_classifier
        self.__faces_data = []  # Liste qui contiendra sous forme de dictionnaire l'ensemble des données liés aux différentes images trouvées.

    def __get_faces_data(self, img, with_best_emotion_model):
        """
        Récupère les données à partir de l'image fournis en entrée.
        :param img: Image d'où on veut récupérer les info concernant les visages.
        :param with_best_emotion_model: Permet de choisir si oui ou non on veut le meilleur modèle pour analyser les émotion.
            Si oui il est lent mais efficace (donc pas utilisable pour la vidéo), si non, il est rapide pas assez efficace sur des images.
        """
        self.__faces_data = []
        # Multi scale signifie que l'algo va passer plusieurs fois dans des sous régions
        #   afin de détecter des visages de taille variantes.
        rectangle_detected = self.faces_recognition_classifier.detectMultiScale(
            img,
            scaleFactor=1.1,  # % de réduction de la taille de l'image.
            minNeighbors=5,  # Nombre min. de rectangle voisin dans une section pour valider que c'est bien un visage.
            minSize=(30, 30)  # Taille min du rectangle.
        )
        emotion_model = FER(
            mtcnn=with_best_emotion_model)  # Model qui permet de reconnaître l'émotion d'un seul visage.
        # Récupère à partir du modèle les coodonnées des visages détectés, où x|y sont les coordonnées du premier point
        #       du rectangle, et w|h (width et height) sont les distances par rapport à ces deux points.
        for (x, y, w, h) in rectangle_detected:
            faces_coordinates = {
                'rectangle_start_coord_x': x,
                'rectangle_start_coord_y': y,
                'rectangle_end_coord_x': x + w,
                'rectangle_end_coord_y': y + h
            }
            face = img[y: y + h,
                   x: x + w]  # Récupère, à partir des coordonnées x|y, les visages de l'image passé en entrée "img".
            mood = 'Mask' if (face_is_masked(face)) else get_top_mood(face, emotion_model)  # Commence par détecter si le visage est masqué, sinon détecte son émotion.
            if mood is None:  # Ignore les coordonnées qui ne sont pas un visage dans le cas où ce n'est ni un visage masqué, ni un visage avec des émotions.
                print('pas de mood trouvé')
                continue

            self.__faces_data.append({  # Liste de visages détectés sous forme de dictionnaires contenants :
                'mood': mood,  # L'émotion du visages détectée.
                'face': face,  # Le visages sous forme d'image détecté.
                'face_coord': faces_coordinates  # Les coordonnées du visage.
            })

    def __place_emoji(self, img):
        """
        Place les émojis sur l'image fournie à partir des données receuillis précédements par la fonction "__Get_faces_data".
        :param img: Image où l'on veut y placer les émojis.
        """
        for face_data in self.__faces_data:
            # Récupère les coordonnées du visage à une itération n.
            faces_x1_point = face_data['face_coord']['rectangle_end_coord_x']
            faces_x2_point = face_data['face_coord']['rectangle_start_coord_x']
            faces_y1_point = face_data['face_coord']['rectangle_end_coord_y']
            faces_y2_point = face_data['face_coord']['rectangle_start_coord_y']
            # Récupère les dimensions de ce visage afin de pour redimensionner l'émoji à la taille correcte.
            width = faces_x1_point - faces_x2_point
            height = faces_y1_point - faces_y2_point
            img[faces_y2_point: faces_y1_point, faces_x2_point: faces_x1_point] = get_emoji(face_data['mood'], (
                width, height))  # Récupère l'émoji correcpondante au visage et le remplace dans l'image.

    def get_faces(self):
        """
        Retourne les visages détectés.
        :return: Une liste d'images correspondant aux visages trouvés par le modèle.
        """
        return [f['face'] for f in self.__faces_data]

    def get_image_with_faces(self, image, with_best_emotion_model=True):
        """
        Retourne l'image fourni en entrée avec les émoji placé dessus.
        :param image: Image à analyser et où l'on veut placer les émoji.
        :param with_best_emotion_model: Permet de choisir si oui ou non on veut le meilleur modèle pour analyser les émotion.
            Si oui il est lent mais efficace (donc pas utilisable pour la vidéo), si non, il est rapide pas assez efficace sur des images.
        :return: L'image avec les émojis placées à l'endroit où se trouve les visages.
        """
        if type(image) == type(''):
            image = cv.imread(image)
        img_with_rectangle = copy.deepcopy(
            image)  # Copie l'image pour pouvoir faire toutes les opérations sans affecter celle-ci.
        self.__get_faces_data(img_with_rectangle,
                              with_best_emotion_model)  # Récupère et stocke dans une liste l'ensemble des info lié aux visages trouvés.
        self.__place_emoji(
            image)  # Place les émoji sur l'image fourni grâce aux informations receuillies par al fonction au dessus.
        return image

    def play_video(self, with_emoji=True, video=0):
        """
        Permet de faire la même chose que sur une image mais via les frame de la caméra.
        :param with_emoji: Permet de choisir si on veut placer des émoji sur l'image ou un carré rouge sur le visage.
        :param video: port de la vidéo, normalement on doit pas y toucher.
        """
        cap = cv.VideoCapture(video)
        while True:
            ret, frame = cap.read()  # frame = image à un instant t de la caméra, ret = si la caméra est activé/connecté.
            if not ret:
                print("Caméra introuvable ou non branchée")
                break
            # ----------------------------------------------------------------------------  La partie ci-dessous fait le placement d'émoji comme sur une image classique
            if (
                    with_emoji):  # Permet de sélectionner l'option de placement d'émoji où dans le cas contraire d'un carré rouge.
                frame = self.get_image_with_faces(frame, False)
            # ----------------------------------------------------------------------------  La partie ci-dessous créer un carré rouge autour du visage (au cas où ce serait trop lent de faire le placement emoji)
            else:
                rgb_frame = frame[:, :, ::-1]  # Convertis en RBG pour pouvoir être utilisé
                face_locations = face_recognition.face_locations(rgb_frame)  # Trouve les visages dans l'image

                for top, right, bottom, left in face_locations:
                    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv.imshow('Video', frame)
            if cv.waitKey(25) == 13:  # Wait for Enter key to stop (espace)
                break


if __name__ == "__main__":
    """
    p = PlaceEmoji(direcotry_path + 'Perso\\' + 'gm1.jpg')
    p.play_video('test2.mp4')
    """
    img_name = 'gm4.jpg'
    # picture_name = "Training_3908.jpg"
    test_image = directory_path + 'Perso\\' + img_name

    pe = PlaceEmoji(test_image)
    result = pe.get_image_with_faces()

    if result.shape[0] > 900 or result.shape[1] > 1500:
        result = resize_image(result, (1280, 760))
    cv.imshow("Faces found", result)
    cv.waitKey(0)
