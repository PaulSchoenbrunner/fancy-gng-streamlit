import numpy as np
import constants # Importiert erstelltes constants-Modul, das Konstanten wie Mittelwerte und Standardabweichungen enthält
from fancy_pca import FancyPCA # Importiert die FancyPCA-Klasse aus fancy_pca.py

# Importiert die gaussian_filter-Funktion von scipy, um Gaußsche Glättung durchzuführen -> g(x) = 1/(sqrt(2*pi)*sigma) * exp(-x^2/(2*sigma^2))
from scipy.ndimage import gaussian_filter 


#Hier werden die Farben basierend auf einer PCA verändert und die Cluster geglättet

# Unbrauchbar, da die Funktion nicht verwendet wird
def _modify_eig_values(eig_values, alphas):
    eig_values_list = list() # Initialisiert eine leere Liste für die veränderten Eigenwerte
    for alpha in alphas:
        eig_values_list.append(eig_values *  alpha) # Multipliziert die Eigenwerte mit dem aktuellen Alpha-Wert und fügt sie der Liste hinzu
    return eig_values_list # Gibt die Liste der veränderten Eigenwerte zurück


# Aus dem Fancy-PCA-Paper bzw. Fancy-GNG-Paper
def _fancy_pca_vectors(data):
    standard_deviation = constants.FANCY_PCA_STANDARD_DEVIATION
    mean = constants.FANCY_PCA_MEAN
    mean_free_data = data - np.mean(data) # Subtrahiert den Mittelwert von den Daten (zentriert die Daten)
    data_covarianz = np.cov(mean_free_data, rowvar=False) # Berechnet die Kovarianzmatrix der Daten, rowvar=False -> Daten sind in Spalten gespeichert
    eig_values, eig_vecs = np.linalg.eigh(data_covarianz) # Berechnet die Eigenwerte und Eigenvektoren der Kovarianzmatrix, eigh -> Eigenwerte sind sortiert nach Größe
    #alphas =  np.random.normal(mean, standard_deviation, cluster_count) # Pauls Code: Erzeugt zufällige Alpha-Werte aus einer Normalverteilung
    alphas = np.random.randn(3) * standard_deviation + mean # Berichtigter Code: Erzeugt zufällige Alpha-Werte aus einer Normalverteilung
    #alphas = self.standard_deviation*np.random.randn(3)+self.mean
    #alphas = np.array([-0.7, 0.7, -0.3]) -> Debugging
    #eig_values_list = _modify_eig_values(eig_values, alphas) # Modifiziert die Eigenwerte mit den Alpha-Werten, um die Farben bzw. PCA-Transformationen zu variieren
    eig_values_list = eig_values * alphas
    final_vecs = eig_vecs @ eig_values_list # Multipliziert Eigenvektoren mit veränderten Eigenwerten (Rechnung mathematisch gepfrüft mit Olli: ist richtig), @ -> Matrixmultiplikation
    #print(final_vecs)
    #print(eig_values)
    #p_mat = np.array(eig_vecs) # Wandelt die Eigenvektoren in ein numpy-Array um
    #final_vecs = list() # Initialisiert eine leere Liste für die finalen Vektoren
    #for eig_values in eig_values_list: # Schleife über alle veränderten Eigenwerte
    #    final_vecs.append(np.dot(p_mat, eig_values)) # Multipliziert die Eigenvektoren mit den veränderten Eigenwerten und fügt sie der Liste hinzu, np.dot -> Matrixmultiplikation
    return final_vecs # Gibt die berechneten Vektoren zurück

    
def modify_clusters(data, pixel_cluster_map, cluster_count, size_images, data_index):
    data_modify = data.copy() # Kopiert die Eingabedaten, um sie zu ändern, ohne das Original zu verändern
    #print("Debug2:", data.shape)
    #print(data_modify.shape)
    add_vecs = list() # Initialisiert eine leere Liste für die Vektoren, die für die Farbänderung verwendet werden
    #print(cluster_count)
    for i in range(cluster_count):
        add_vecs.append(_fancy_pca_vectors(data_modify[pixel_cluster_map == i])) # Berechnet die veränderten PCA-Vektoren für die Farbänderung
    #print('danach', data_modify.shape)
    add_vecs_smooth = _smooth_add_vecs(pixel_cluster_map, size_images, add_vecs, data_index) # Glättet die Vektoren basierend auf den Clusterinformationen
    data_modify += add_vecs_smooth # Addiert die geglätteten Vektoren zu den ursprünglichen Daten hinzu
    clipped_data = np.clip(data_modify, 0, 1) # Beschränkt die Daten auf den Bereich [0, 1] (Klippt Werte außerhalb dieses Bereichs)
    return clipped_data # Gibt die bearbeiteten Daten zurück

def _smooth_add_vecs(pixel_cluster_map, size_images, add_vecs, data_index):
    vector_field = list() # Initialisiert eine leere Liste für die Vektorfeld-Daten (Vektoren für jedes Pixel)
    #print(add_vecs)
    for cluster_idx in pixel_cluster_map: # Schleife über alle Cluster-Indizes im Cluster-Mapping
        vector_field.append(add_vecs[int(cluster_idx)]) # Fügt den Vektor für das aktuelle Cluster hinzu
    vector_field = np.array(vector_field) # Wandelt die Liste der Vektoren in ein numpy-Array um
    #print(vector_field.shape)

    # Reshape das Vektorfeld in die Form, die mit den Bilddimensionen übereinstimmt, size_images[data_index][1] -> Höhe, size_images[data_index][0] -> Breite, 3 -> RGB-Kanäle
    vector_field = vector_field.reshape((size_images[data_index][1], size_images[data_index][0], 3)) 
    
    if constants.USE_SMOOTH: # Überprüft, ob das Glätten aktiviert ist (True)
        print("Smooth")
        smoothed_vector_field = np.zeros_like(vector_field) # Erstellt ein Array mit der gleichen Form wie das Vektorfeld, aber mit Nullen gefüllt

        for i in range(3): # Schleife über die 3 Farbkanäle (RGB)

            # Glättet den Vektor für jeden Farbkanal mit einem Gauß-Filter, sigma=constants.SIGMA -> Standardabweichung des Gauß-Filters, [:, :, i] -> Zugriff auf den i-ten Farbkanal
            smoothed_vector_field[:, :, i] = gaussian_filter(vector_field[:, :, i], sigma=constants.SIGMA)
    else:
        smoothed_vector_field = vector_field # Wenn das Glätten nicht aktiviert ist, wird das originale Vektorfeld verwendet
    
    return smoothed_vector_field.reshape(-1,3) # Gibt das geglättete Vektorfeld als 1D-Array zurück, reshape(-1,3) -> -1 -> automatische Berechnung der Zeilen, 3 -> 3 Spalten (Farbkanäle RGB) 






 