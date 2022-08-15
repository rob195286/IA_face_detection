import cv2 as cv
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from images_management import directory_path


prototxtPath = "Model\\" + "deploy.prototxt" # config du réseau
weightsPath = "Model\\" + "res10_300x300_ssd_iter_140000.caffemodel"# modèle pré-entrainé permettant de charger les poids du réseau
net = cv.dnn.readNet(prototxtPath, weightsPath) # initialisation du réseau

model = load_model("Model\\" + "model.h5")


def face_is_masked(img, confidence=0.8):
    """
    Permet de renvoyer "True" si le visages à un masque, "False" sinon.
    :param img: Image contenant le visage à analyser.
    :param confidence: Niveau de confiance minimum pour renvoyer "True" en %.
    :return: "True" si le visage à un masque, "False" sinon.
    """
    image = img.copy()
    blob = cv.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0)) # un blob est une image qui a été préprocessé, préparer pour travailler dessus
    net.setInput(blob) # passe l'image préprocessé au réseau.
    detections = net.forward() # exécute un passage avant en calcul la sortie de chaque couche intermédiaire.

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2] # % de confiance actuel que c'est un visage.
        if conf > confidence:
            face = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            face = cv.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0] # détermine si c'est masqué ou pas.
            return True if mask > withoutMask else False


if __name__ == "__main__":
    test_image = directory_path + 'Mask\\with_mask\\' + '0_0_0 copy 5.jpg'
    result = face_is_masked(cv.imread(test_image))
    print(result)

