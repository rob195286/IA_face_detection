# IA_face_detection
## Installation
Pour pour faire fonctionner le projet, il est nécessaire d'avoir les librairies suivantes :
  * opencv -> https://docs.opencv.org/4.x/index.html.
  * FER -> https://github.com/justinshenk/fer#installation.

## Utilisation
La façon d'utiliser le projet se fait via une ligne de commande en lancant le fichier __main__.py et en fournissant le chemin où est stocké l'image à analyser. 
L'exemple suivant montre comment l'utiliser : </br> </br>
   python __main__.py -p 'chemin_absolu_image'
   
Pour faire une analyse video, utiliser cette commande :
   python __main__.py -p 'chemin_absolu_de_la_video' -v True

Ensuite l'image fournie s'affichera en utilisant opencv avec des émotes sur les visages en correspondance de leurs émotions.
