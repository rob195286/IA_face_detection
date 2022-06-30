import cv2
from images_management import direcotry_path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np



prototxtPath = "Model\\" + "deploy.prototxt"
weightsPath = "Model\\" + "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

model = load_model("Model\\" + "model.h5")

def face_is_masked(img, confidence = 0.8):
    image = img.copy()

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0)) # un blob est une image qui a été préprocessé, préparer pour travailler dessus
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > confidence:
            face = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]
            print('mask : ', mask)
            print('withoutMask : ', withoutMask)
            return True if mask > withoutMask else False


if __name__ == "__main__" :
    img_name = '0_0_0 copy 5.jpg'
    test_image = direcotry_path + 'Mask\\with_mask\\' + img_name
    result = face_is_masked(cv2.imread(test_image))
    print(result)

