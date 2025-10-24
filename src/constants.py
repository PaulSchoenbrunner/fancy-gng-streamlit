#Hyperparameter die können auf die Bildgrößen/arten optimiert werden

#GNG

#Anzahl Epochen für GNG
EPOCH = 10
#Kanten zwischen Gruppen entfernen
EDGE_CUTTING = 0.65
#Anzahl der Startnodes
SARTING_NODES = 3 ### Schreibfehler: STARTING_NODES ### -> überprüfe auch in anderen files und passe ggf. alle an
#Anzahl der Maximalen Nodes
MAX_NODES = 20
#Gaussglättung verwenden
USE_SMOOTH = True

#Data
#MAX RGB-Value
MAX_COLOR_VALUE = 255.0
#Art der Inputbilder
FILE_TYPE = ".jpg" ### HINWEIS: Groß- und Kleinschreibung wichtig für Mac/Linux ###
#Anzahl der generierten Bilder
AUG_COUNT = 5

#Fancy-PCA
#Mittelwert
FANCY_PCA_MEAN = 0
#Standardabweichung für die alpha Ziehung. Je größer die STD, umso stärker die Veränderung
FANCY_PCA_STANDARD_DEVIATION = 5 ### Debugging: setze auf 5 für starke Änderung (normalerweise auf 1 gesetzt) -> siehe z.B. 'alpha' bei Input 'blumenwiese2.jpg' ###


#Gaussian
#Stärke der Glättung
SIGMA = 3


#Color-jitter
BRIGHTNESS = 0.4
CONTRAST = 0.4
SATURATION = 0.4
HUE = 0.1
