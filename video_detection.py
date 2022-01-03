import cv2
import face_recognition



# https://medium.com/analytics-vidhya/face-detection-on-recorded-videos-using-opencv-in-python-windows-and-macos-407635c699
cap = cv2.VideoCapture("test.mp4")

face_locations = []

if __name__ == "__main__" :
    while True:
        _, frame = cap.read() # prend une frame, la première valeur indique si elle a été prise, pas important pour nous.
        rgb_frame = frame[:, :, ::-1] # Convertis en RBG pour pouvoir être utilisé
        face_locations = face_recognition.face_locations(frame) # Trouve les visages dans l'image

        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) == 13: # Wait for Enter key to stop
            break