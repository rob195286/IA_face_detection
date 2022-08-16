# IA_face_detection
## Installation
Pour pour faire initialiser le projet, installer les requirements.

Les bibiliothèques utiliséés sont :
* numpy
* tensorflow
* opencv-python==4.5.5.64
* fer
* cmake
* face_recognition

## Utilisation
La façon d'utiliser le projet se fait via une ligne de commande en lancant le fichier __main__.py et en fournissant le chemin où est stocké l'image à analyser. 
L'exemple suivant montre comment l'utiliser : </br> </br>
   python __main__.py -p 'chemin_absolu_image'
   
Pour faire une analyse video à partir d'une caméra, utiliser cette commande : </br> </br>
   python.exe __main__.py -v True

Ensuite l'image fournie s'affichera en utilisant opencv avec des émotes sur les visages en correspondance de leurs émotions.
