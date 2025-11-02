import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import io, zipfile
import datetime
import fancy_pca as FP


#----------------------------UI Constants-------------------------------------------------------
FANCYGNG_STR = "FancyGNG"
FANCYPCA_STR = "FancyPCA"
COLORJITTER_STR = "Color-Jitter"
MAX_UI_AUG_COUNT = 10


#-----------------------------Session------------------------------------------------------------
def init_session():
    # Session-States initialisieren
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    if "image_results" not in st.session_state:
        st.session_state.image_results = {}  # {filename: {"original": Image, "aug_images": [...], "cluster_count": int, "data_shape": tuple}}
    if "fig_png" not in st.session_state:
        st.session_state.fig_png = {}
    if "done" not in st.session_state:
        st.session_state.done = False
    if "last_picture" not in st.session_state:
        st.session_state.last_picture = None
    if "last_aug" not in st.session_state:
        st.session_state.last_aug = None
    if "last_aug_info" not in st.session_state:
        st.session_state.last_aug_info = None

init_session()

def reset_session():
    st.session_state.clear()
    init_session()

def reset_for_new_run():
    st.session_state.image_results = {}
    st.session_state.fig_png = {}



#--------------------------Streamlit UI-------------------------------
st.title("🧠 DBL-GNG Image Augmentation")
st.write("Lade ein oder mehrere Bilder hoch oder nimm eines mit der Kamera auf.")


# Augementation wählen
aug_option = st.selectbox(
    "Wähle das Augmentationsverfahren:",
    [FANCYGNG_STR, FANCYPCA_STR, COLORJITTER_STR],
    index=0
)
st.write(f"Gewähltes Verfahren: {aug_option}")


# Quelle wählen
input_option = st.radio("Bildquelle auswählen:", ["Datei-Upload", "Kamera"])

if input_option == "Datei-Upload":
    uploaded_files = st.file_uploader(
        "Bilder auswählen", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        on_change= reset_session
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

elif input_option == "Kamera":
    camera_image = st.camera_input("Bild aufnehmen")
    if camera_image is not None:

        if st.session_state.last_picture is None:
            reset_session()

        elif st.session_state.last_picture is not None and camera_image.getvalue() != st.session_state.last_picture:
            reset_session()
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        camera_image.name = f"camera_{timestamp}.jpg"        
        st.session_state.uploaded_files = [camera_image]
        st.session_state.last_picture = camera_image.getvalue()      
        
       
        
#Punktwolke anzeigen
show_point_cloud = st.checkbox("Zeige die Punktwolke")

#Punktwolke anzeigen
show_gray_scale = st.checkbox("Generiere zusätzlich eine gray scale Version")

option_buttons = []
option_buttons.append(show_point_cloud)
option_buttons.append(show_gray_scale)
#Augmentation starten
start_augmentation = st.button("🚀 Starte Augmentierung")
if start_augmentation and st.session_state.done:
    reset_for_new_run()


#----------------------------Sidebar für Parameter---------------------------------
st.sidebar.header("⚙️ Parameter-Einstellungen")

# Allgemeine Parameter (für alle Methoden)
st.sidebar.subheader("Allgemein")
AUG_COUNT = st.sidebar.number_input("Anzahl der Augmentationen", min_value=1, max_value=100,  value=getattr(constants, "AUG_COUNT", 3))

# Dynamische Sektionen je nach ausgewählter Methode
if aug_option == COLORJITTER_STR:
    st.sidebar.subheader("🧮 Color Jitter Parameter")
    BRIGHTNESS = st.sidebar.slider("Helligkeit", 0.0, 2.0, getattr(constants, "BRIGHTNESS", 0.5))
    CONTRAST = st.sidebar.slider("Kontrast", 0.0, 2.0, getattr(constants, "CONTRAST", 0.5))
    SATURATION = st.sidebar.slider("Sättigung", 0.0, 2.0, getattr(constants, "SATURATION", 0.5))
    HUE = st.sidebar.slider("Farbton", 0.0, 0.5, getattr(constants, "HUE", 0.1))

    # Werte übernehmen
    constants.BRIGHTNESS = BRIGHTNESS
    constants.CONTRAST = CONTRAST
    constants.SATURATION = SATURATION
    constants.HUE = HUE
    
    

elif aug_option == FANCYGNG_STR:
    st.sidebar.subheader("🧮 Fancy GNG Parameter")
    STANDARD_DEVIATION = st.sidebar.slider("Standard Abweichung", 1, 10, getattr(constants, "FANCY_PCA_STANDARD_DEVIATION", 20))
    MEAN = st.sidebar.slider("Mittelwert", 0, 10, getattr(constants, "FANCY_PCA_MEAN", 3))  
    USE_SMOOTH = st.sidebar.checkbox("Nutze Glättung", value=True)
    constants.FANCY_PCA_STANDARD_DEVIATION = STANDARD_DEVIATION
    constants.FANCY_PCA_MEAN = MEAN
    constants.USE_SMOOTH = USE_SMOOTH
    


elif aug_option == FANCYPCA_STR:
    st.sidebar.subheader("🧮 Fancy PCA Parameter")
    STANDARD_DEVIATION = st.sidebar.slider("Standard Abweichung", 0, 10, getattr(constants, "FANCY_PCA_STANDARD_DEVIATION", 20))
    MEAN = st.sidebar.slider("Mittelwert", 0, 10, getattr(constants, "FANCY_PCA_MEAN", 3))  
    constants.FANCY_PCA_STANDARD_DEVIATION = STANDARD_DEVIATION
    constants.FANCY_PCA_MEAN = MEAN
    
constants.AUG_COUNT = AUG_COUNT

print(constants.FANCY_PCA_STANDARD_DEVIATION, constants.FANCY_PCA_MEAN, constants.USE_SMOOTH)
#-----------------------------------FancyPCA------------------------------------------
def fancy_pca(image_data, original_iamge):
    aug_images = generate_fancy_pca_augmentations(image_data)
    st.session_state.image_results[filename] = {
                   "original": original_iamge,
                   "aug_images": aug_images,
                   "data_shape": image_data.shape,
    }



def generate_fancy_pca_augmentations(image_data):
    fancy_pca_transform = FP.FancyPCA()
    aug_images = []
    for _ in range(constants.AUG_COUNT):
         # Wendet Fancy PCA auf das Bild an (Kopie wird verwendet, um Original nicht zu überschreiben, so wie bei fancy_pca_runner.py)
        fancy_pca_image = fancy_pca_transform.fancy_pca(image_data.copy())
        # Umwandlung auf den Bereich [0, 255] und zu uint8
        fancy_pca_image = (fancy_pca_image * 255).astype(np.uint8)
        # Rückumwandlung in die ursprüngliche Bildform: Höhe x Breite x 3
        height, width, channels = image.size[1], image.size[0], 3
        fancy_pca_image = fancy_pca_image.reshape((height, width, channels))
        try:
            aug_image = Image.fromarray(fancy_pca_image)
            aug_images.append(aug_image)
        except Exception as e:
            st.write(f"Fehler bei der Bildumwandlung: {e}")  ### Debugging ###

    return aug_images

def show_fancy_pca_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Bildgröße:** {info['original'].size}")
    st.write(f"**Bildarray-Form:** {info['data_shape']}")





#-------------------------------------FancyGNG---------------------------------------------
def fancy_gng(image_data, original_image):
    aug_images, cluster_count = generate_fancy_gng_augmentations(image_data)
    st.session_state.image_results[filename] = {
                   "original": original_image,
                   "aug_images": aug_images,
                   "cluster_count": cluster_count,
                   "data_shape": image_data.shape,
    }


def show_fancy_gng_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Bildgröße:** {info['original'].size}")
    st.write(f"**Bildarray-Form:** {info['data_shape']}")
    st.write(f"**Anzahl der Cluster:** {info['cluster_count']}")


def generate_fancy_gng_augmentations(image_data):
    """Führt den gesamten DBL-GNG + Clustering + Augmentierungsprozess durch."""
    gng = dbl_gng.DBL_GNG(3, constants.MAX_NODES)
    gng.initializeDistributedNode(image_data, constants.SARTING_NODES)
    bar = trange(constants.EPOCH)
    for i in bar:
        gng.resetBatch()
        gng.batchLearning(image_data)
        gng.updateNetwork()
        gng.addNewNode(gng)
        bar.set_description(f"Epoch {i + 1} Knotenanzahl: {len(gng.W)}")  # Fortschrittsanzeige

    gng.cutEdge()
    gng.finalNodeDatumMap(image_data)

    finalDistMap = gng.finalDistMap
    finalNodes = (gng.W * constants.MAX_COLOR_VALUE).astype(int)
    connectiveMatrix = gng.C

    pixel_cluster_map, node_cluster_map = clustering.cluster(finalDistMap, finalNodes, connectiveMatrix)
    pixel_cluster_map = np.array(pixel_cluster_map)
    cluster_count = int(max(node_cluster_map)) + 1

    aug_images = []
    for _ in range(constants.AUG_COUNT):
        aug_data = color_pca.modify_clusters(data_array, pixel_cluster_map, cluster_count, [image.size], 0)
        aug_data = (aug_data * 255).astype(np.uint8)  # Umwandlung in uint8

        # Rückwandlung in die ursprüngliche Bildform: Höhe x Breite x 3
        aug_data = aug_data.reshape((image.size[1], image.size[0], 3))

        # Erstellen des augmentierten Bildes
        aug_image = Image.fromarray(aug_data)
        aug_images.append(aug_image)
    return aug_images, cluster_count



#-----------------------------------Color-Jitter------------------------------------------
def color_jitter(image_data, original_image):
    aug_images = generate_color_jitter_augmentations(original_image)
    st.session_state.image_results[filename] = {
        "original": original_image,
        "aug_images": aug_images,
        "data_shape": image_data.shape,
        "parameter": {"Brightness" : constants.BRIGHTNESS,
                      "Contrast" : constants.CONTRAST,
                      "Saturation" : constants.SATURATION,
                      "Hue" : constants.HUE}
    }


def generate_color_jitter_augmentations(image):
    transform = transforms.ColorJitter(
        brightness=constants.BRIGHTNESS, contrast=constants.CONTRAST, saturation=constants.SATURATION, hue=constants.HUE
    )
    aug_images = []
    for _ in range(constants.AUG_COUNT):
        img = transform(image)
        aug_images.append(img)
    return aug_images


def show_color_jitter_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Bildgröße:** {info['original'].size}")
    st.write(f"**Bildarray-Form:** {info['data_shape']}")
    st.write(f"**Parameter:** {info['parameter']}")





#-----------------------------Plotting----------------------------------------------
def create_point_cloud(all_images, axs, row_idx = 0):
    for idx, img in enumerate(all_images):
        ax = get_fig_ax(axs, row_idx, idx)
        if len(ax.images) == 0 and len(ax.collections) == 0:  # nur wenn Achse leer
            rgb_image = img.convert("RGB")
            width, height = img.size
            points = np.array([
                (r, g, b, r, g, b)
                for x in range(width)
                for y in range(height)
                for (r, g, b) in [rgb_image.getpixel((x, y))]
            ])
            ax.scatter(points[:, 1], points[:, 2], c=points[:, 3:6] / 255, s=1)
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect('equal', 'box')
            #ax.set_title("Original" if idx == 0 else f"Aug {idx}")
        

def create_gray_images(all_images, axs, row_idx = 0):
    grayscale_transform = transforms.Grayscale()
    for idx, img in enumerate(all_images):
        ax = get_fig_ax(axs, row_idx, idx)
        if len(ax.images) == 0 and len(ax.collections) == 0:
            gray = grayscale_transform(img)
            ax.imshow(gray, cmap="gray")
            ax.axis("off")
            #ax.set_title("Original" if idx == 0 else f"Gray Aug {idx}")


def create_main_plot(all_images, axs, row_idx = 0):
    for idx, img in enumerate(all_images):
        ax = get_fig_ax(axs, row_idx, idx)
        if len(ax.images) == 0 and len(ax.collections) == 0:  
            ax.imshow(img)
            ax.axis("off")
            ax.set_title("Original" if idx == 0 else f"Aug {idx}")


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf






#---------------------------Hilfsfunktion und Variablen-------------------------------------------------

def keep_dependent_ui_element_at_random_button(dependency, dependency_func_map : dict):
    if dependency is not None: 
        entry = dependency_func_map.get(dependency)
        if entry and len(entry) > 0:
            func = entry[0]
            args = entry[1:]
            return func(*args)  

def get_fig_ax(axs, row_index, idx):
    if axs.ndim == 1:
        return axs[idx]
    else:
        return axs[row_index, idx]

#show_fancy_pca_info_bool = (aug_option == FANCYPCA_STR and start_augmentation) or (aug_option == FANCYGNG_STR and st.session_state.last_aug == FANCYPCA_STR and not start_augmentation)
#show_fancy_gng_info_bool = (aug_option == FANCYGNG_STR and start_augmentation) or (aug_option == FANCYPCA_STR and st.session_state.last_aug == FANCYGNG_STR and not start_augmentation)






#------------------------------------------Hauptverarbeitung----------------------------------------------------------------------------------
if (start_augmentation or st.session_state.done) and st.session_state.uploaded_files:
    for uploaded_file in st.session_state.uploaded_files:
        filename = uploaded_file.name

        # Falls bereits berechnet, überspringen
        if filename not in st.session_state.image_results and start_augmentation:
            st.session_state.last_aug = aug_option
            with st.spinner(f"Verarbeite {filename} ..."):
                image = Image.open(uploaded_file).convert("RGB")
                image_array = np.asarray(image)
                data_array = image_array.reshape(-1, 3) / constants.MAX_COLOR_VALUE

                if aug_option == FANCYGNG_STR:
                    fancy_gng(data_array, image)
                
                elif aug_option == FANCYPCA_STR:
                    fancy_pca(data_array, image)

                elif aug_option == COLORJITTER_STR:
                    color_jitter(data_array, image)

                
                

        # Info Anzeige
        info = st.session_state.image_results[filename]
       
        if filename not in st.session_state.fig_png:
            if aug_option == FANCYGNG_STR:
                st.session_state.last_aug_info = show_fancy_gng_info
           
            elif aug_option == FANCYPCA_STR:
                st.session_state.last_aug_info = show_fancy_pca_info

            elif aug_option == COLORJITTER_STR:
                st.session_state.last_aug_info = show_color_jitter_info
        
        st.session_state.last_aug_info(filename, info)
        

        #Grafik
        with st.spinner(f"Augementation von {filename} abgeschlossen...Starte Visualisierung"):
            if filename not in st.session_state.fig_png:
                ax_counter = sum(1 for opt in option_buttons if opt) + 1

                fig, axs = plt.subplots(ax_counter, constants.AUG_COUNT + 1, figsize=(15, 6))


                current_row = 0
                # Punktwolke & Augmentierungen generieren
                if show_point_cloud:
                    create_point_cloud([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1

                #Gray scal ebild generieren
                if show_gray_scale:
                    create_gray_images([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1
                #main fig
                create_main_plot([info["original"]] + info["aug_images"], axs, current_row)

                png_buf = fig_to_png(fig)

                fig.tight_layout()

                st.session_state.fig_png[filename] = png_buf.getvalue()
            


            
      

        #Grafik anzeigen
        st.image(st.session_state.fig_png[filename])
    st.session_state.done = True
        




#--------------------------------------------Download-Bereich----------------------------------------------
if st.session_state.image_results:
    st.divider()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for filename, info in st.session_state.image_results.items():
            base_name = filename.rsplit('.', 1)[0]
            for i, img in enumerate(info["aug_images"]):
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                zipf.writestr(f"{base_name}_aug_{st.session_state.last_aug}_{i+1}.jpg", buf.getvalue())

    download = st.download_button(
        label="⬇️ Augmentierte Bilder als ZIP herunterladen",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip",
    )

    



