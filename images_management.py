from enum import Enum
import os

direcotry_path = os.getcwd() + "\images\\"

class Mood(Enum):
    angry = "angry"
    disgust = "disgust"
    fear = "fear"
    happy ="happy"
    neutral = "neutral"
    sad = "sad"
    surprise = "surprise"

    def __str__(self):
        return '{0}'.format(self.value)


path_images_with_mood = direcotry_path + "Images_with_mood\\train\\" + str(Mood.angry) + "\\"
emoji_path = direcotry_path + "Emoji\\"

