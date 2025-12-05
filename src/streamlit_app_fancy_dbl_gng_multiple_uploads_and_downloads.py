import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
import io, zipfile
import datetime
import fancy_pca as FP
import random



#----------------------------UI Constants-------------------------------------------------------
FANCYGNG_STR = "Fancy-GNG"
FANCYPCA_STR = "Fancy-PCA"
COLORJITTER_STR = "Color-Jitter"
MAX_UI_AUG_COUNT = 10
MAX_UI_AUG_COUNT += 1
CLOUD_SIZE = 5000
REDUCED_TRAINING = 5000


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
    if "gray_images" not in st.session_state:
        st.session_state.gray_images = {}

init_session()

def reset_session():
    st.session_state.clear()
    init_session()

def reset_for_new_run():
    st.session_state.image_results = {}
    st.session_state.fig_png = {}
    st.session_state.gray_images = {}



#--------------------------Streamlit UI-------------------------------
st.title("🧠 Fancy-GNG image augmentation")
st.write("Upload one or more images or take one with your camera.")


# Augementation wählen
aug_option = st.selectbox(
    "Select the augmentation methode:",
    [FANCYGNG_STR, FANCYPCA_STR, COLORJITTER_STR],
    index=0,
    help="Select the augmentation method for generating the images."
)
st.write(f"Method chosen: {aug_option}")


# Quelle wählen
input_option = st.radio("Select image source:", ["File upload", "Camera"],
                help="Select the source of the images to be used for augmentation. " \
                "When selecting the camera, only one image can be captured. When selecting file upload, multiple images can be selected.")

if input_option == "File upload":
    uploaded_files = st.file_uploader(
        "Select images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        on_change= reset_session,
        help="Multiple images are accepted. The supported types are: jpg, jpeg and png"
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

elif input_option == "Camera":
    camera_image = st.camera_input("Take a picture",
    help="A webcam is required to take a picture.")

    if camera_image is not None:

        if st.session_state.last_picture is None:
            reset_session()

        elif st.session_state.last_picture is not None and camera_image.getvalue() != st.session_state.last_picture:
            reset_session()
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        camera_image.name = f"camera_{timestamp}.jpg"        
        st.session_state.uploaded_files = [camera_image]
        st.session_state.last_picture = camera_image.getvalue()      
        
       
option_buttons_ui = []     
#Punktwolke anzeigen
show_point_cloud = st.checkbox("Show the point cloud",
                    help="This option generates a point cloud representation of each image's pixels in the G-B color space. " \
                    "Each point represents one pixel of the image.The position of the points is based on the green and blue values (x=G, y=B). " \
                    "The color of the points corresponds to the original color of the pixel (R,G,B). " \
                    "The result is a 2D visual representation of the color distribution of an image.")

#gray scale anzeigen
show_gray_scale = st.checkbox("Additionally generate a grayscale version",
                    help="Additionally a grayscale version of the augmented image is generated. These images can also be downloaded at the end.")



show_cluster = False
reduced_fancy_gng = False
#cluster
if aug_option == FANCYGNG_STR:
    show_cluster = st.checkbox("Generate a pixel cluster map",
                    help="A unique color is selected for each color cluster (connected codebook vectors) found by GNG. " \
                    "All pixels belonging to this cluster are colored in this color. This visualizes the color clusters found in the image.")
    option_buttons_ui.append(show_cluster)
    
    #Reduziertes Fancy-GNG
    reduced_fancy_gng = st.checkbox("Train Fancy-GNG on fewer data points",
                    help="At random (without repetition), select n pixels from the image to be used to train the GNG. " \
                    "This can be used to accelerate Fancy-GNG.")
    


option_buttons_ui.append(show_point_cloud)
option_buttons_ui.append(show_gray_scale)


#Augmentation starten
start_augmentation = st.button("🚀 Start augmentation")
if start_augmentation and st.session_state.done:
    reset_for_new_run()


#----------------------------Sidebar für Parameter---------------------------------
st.sidebar.header("⚙️ Parameter settings",
                    help="Change parameters of the selected augmentation method")

# Allgemeine Parameter (für alle Methoden)
st.sidebar.subheader("General")
AUG_COUNT = 5
AUG_COUNT = st.sidebar.number_input("Number of augmentations", min_value=1, max_value=100,  value=getattr(constants, "AUG_COUNT", 3),
                                    help="The number of augmentations generated per image")

# Dynamische Sektionen je nach ausgewählter Methode
if aug_option == COLORJITTER_STR:
    st.sidebar.subheader("🧮 Color-Jitter parameter")
    BRIGHTNESS = st.sidebar.slider("Brightness", 0.0, 2.0, getattr(constants, "BRIGHTNESS", 0.5),
                help="Varies the image brightness. Values above 1 make the image brighter, values below 1 make it darker.")
    CONTRAST = st.sidebar.slider("Contrast", 0.0, 2.0, getattr(constants, "CONTRAST", 0.5),
                help="Changes the contrast of the image. Higher values increase the difference between light and dark.")
    SATURATION = st.sidebar.slider("Saturation", 0.0, 2.0, getattr(constants, "SATURATION", 0.5),
                help="Changes the color saturation. Low values desaturate the image, high values intensify the colors.")
    HUE = st.sidebar.slider("Hue", 0.0, 0.5, getattr(constants, "HUE", 0.1),
                help="Shifts the color tone of the image. Small values result in subtle color shifts.")

    # Werte übernehmen
    constants.BRIGHTNESS = BRIGHTNESS
    constants.CONTRAST = CONTRAST
    constants.SATURATION = SATURATION
    constants.HUE = HUE
    
    

elif aug_option == FANCYGNG_STR:
    st.sidebar.subheader("🧮 Fancy-GNG parameter")
    STANDARD_DEVIATION = st.sidebar.slider("Standard deviation", 1.0, 10.0,  float(getattr(constants, "FANCY_PCA_STANDARD_DEVIATION", 2.0)),
                    step=0.25,
                    help="Determines the strength of the color shift along the PCA components. Higher values produce stronger color variations.")
    MEAN = st.sidebar.slider("Mean", 0.0, 10.0,  float(getattr(constants, "FANCY_PCA_MEAN", 3.0)),
                    step=0.25,
                    help="Sets the average shift along the color PCA. Affects how much colors are changed on average.")  
    #USE_SMOOTH = st.sidebar.checkbox("Use smoothing", value=False)
    #if USE_SMOOTH:
    #    SIGMA = st.sidebar.slider("Smoothing/Sigma", 0, 10, getattr(constants, "SIGMA", 3))
    #    constants.SIGMA = SIGMA
    
    
    constants.FANCY_PCA_STANDARD_DEVIATION = STANDARD_DEVIATION
    constants.FANCY_PCA_MEAN = MEAN
    #constants.USE_SMOOTH = USE_SMOOTH
    


elif aug_option == FANCYPCA_STR:
    st.sidebar.subheader("🧮 Fancy-PCA parameter")
    STANDARD_DEVIATION = st.sidebar.slider("Standard deviation", 0.0, 10.0,  float(getattr(constants, "FANCY_PCA_STANDARD_DEVIATION", 2.0)),
                        step=0.25,
                        help="Determines the strength of the color shift along the PCA components. Higher values produce stronger color variations.")
    MEAN = st.sidebar.slider("Mean", 0.0, 10.0,  float(getattr(constants, "FANCY_PCA_MEAN", 3.0)),
                        step=0.25,
                        help="Sets the average shift along the color PCA. Affects how much colors are changed on average.")  
    constants.FANCY_PCA_STANDARD_DEVIATION = STANDARD_DEVIATION
    constants.FANCY_PCA_MEAN = MEAN

if show_point_cloud:
    st.sidebar.subheader("☁️ Size of the point cloud")
    CLOUD_SIZE = st.sidebar.number_input("Number of points", 100, 1000000, CLOUD_SIZE,
                help="By default, 5000 random pixels from the image are selected for displaying the point cloud. " \
                "This helps to generate the point cloud faster and save computing power. " \
                "However, any other number smaller than the total number of pixels can also be selected. ")
    use_original_size = st.sidebar.checkbox("Use original image size",
                help="Use all pixels of the image to generate the point cloud. This may take some time for larger images.")
    
if reduced_fancy_gng and aug_option == FANCYGNG_STR:
    st.sidebar.subheader("Number of pixels used for Fancy-GNG training")
    REDUCED_TRAINING = st.sidebar.number_input("Number of pixels", 100, 1000000, REDUCED_TRAINING,
                help="By default, the reduced GNG training uses 5000 random pixels from the image. " \
                "This helps to train GNG faster and save computing power. " \
                "However, any other number that is smaller than the total number of pixels can also be selected. ")
    

constants.AUG_COUNT = AUG_COUNT

#print(constants.FANCY_PCA_STANDARD_DEVIATION, constants.FANCY_PCA_MEAN, constants.USE_SMOOTH)
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
            st.write(f"Error during image conversion: {e}")  ### Debugging ###

    return aug_images

def show_fancy_pca_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Image size:** {info['original'].size}")
    st.write(f"**Image array shape:** {info['data_shape']}")





#-------------------------------------FancyGNG---------------------------------------------
def fancy_gng(image_data, original_image):
    aug_images, cluster_count, pixel_cluster_map, node_cluster_map = generate_fancy_gng_augmentations(image_data)
    st.session_state.image_results[filename] = {
                    "original": original_image,
                    "aug_images": aug_images,
                    "cluster_count": cluster_count,
                    "data_shape": image_data.shape,
                    "pixel_cluster_map": pixel_cluster_map,
                    "nodes": node_cluster_map.size
    }


def show_fancy_gng_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Image size:** {info['original'].size}")
    st.write(f"**Image array shape:** {info['data_shape']}")
    st.write(f"**Number of clusters:** {info['cluster_count']}")
    st.write(f"**Number of codebook vectors generated by GNG:** {info['nodes']}")


def generate_fancy_gng_augmentations(image_data):
    """Führt den gesamten DBL-GNG + Clustering + Augmentierungsprozess durch."""
    gng = dbl_gng.DBL_GNG(3, constants.MAX_NODES)
    image_data_org = image_data.copy()
    #print("Reduce:", len(image_data), image_data.shape)
    if reduced_fancy_gng and REDUCED_TRAINING < len(image_data):
        indices = np.random.choice(len(image_data), REDUCED_TRAINING, replace=False)
        image_data = image_data[indices]
        
    gng.initializeDistributedNode(image_data, constants.SARTING_NODES)
    bar = trange(constants.EPOCH)
    for i in bar:
        gng.resetBatch()
        gng.batchLearning(image_data)
        gng.updateNetwork()
        gng.addNewNode(gng)
        bar.set_description(f"Epoch {i + 1} number of nodes: {len(gng.W)}")  # Fortschrittsanzeige

    gng.cutEdge()
    gng.finalNodeDatumMap(image_data_org)

    finalDistMap = gng.finalDistMap
    
    finalNodes = (gng.W * constants.MAX_COLOR_VALUE).astype(int)
    connectiveMatrix = gng.C

    pixel_cluster_map, node_cluster_map = clustering.cluster(finalDistMap, finalNodes, connectiveMatrix)
    pixel_cluster_map = np.array(pixel_cluster_map)
    print(node_cluster_map.size)
    cluster_count = int(max(node_cluster_map)) + 1


    aug_images = []
    for _ in range(constants.AUG_COUNT):
        aug_data = color_pca.modify_clusters(image_data_org, pixel_cluster_map, cluster_count, [image.size], 0)
        aug_data = (aug_data * 255).astype(np.uint8)  # Umwandlung in uint8

        # Rückwandlung in die ursprüngliche Bildform: Höhe x Breite x 3
        aug_data = aug_data.reshape((image.size[1], image.size[0], 3))

        # Erstellen des augmentierten Bildes
        aug_image = Image.fromarray(aug_data)
        aug_images.append(aug_image)
    return aug_images, cluster_count, pixel_cluster_map, node_cluster_map



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
    st.write(f"**Image size:** {info['original'].size}")
    st.write(f"**Image array shape:** {info['data_shape']}")
    st.write(f"**Parameter:** {info['parameter']}")





#-----------------------------Plotting----------------------------------------------
def create_point_cloud(all_images, axs, row_idx = 0):
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    for idx, img in enumerate(images):
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

             # 🔹 Zufällig 20 000 Punkte auswählen (oder alle, falls weniger)
            if len(points) > CLOUD_SIZE and not use_original_size:
                #print("Capped point cloud")
                indices = np.random.choice(len(points), CLOUD_SIZE, replace=False)
                points = points[indices]

            # Punkte definieren (r,g,b -> als Farbe)
            #points = np.column_stack((pixels, pixels))  # (r,g,b,r,g,b)
            ax.scatter(points[:, 1], points[:, 2], c=points[:, 3:6] / 255, s=1)
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_aspect('equal', 'box')
            #ax.set_title("Original" if idx == 0 else f"Aug {idx}")
        

def create_gray_images(all_images, axs, row_idx = 0):
    images = all_images if len(all_images) <= MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    grayscale_transform = transforms.Grayscale()
    st.session_state.gray_images[filename] = {"images": []}
    for idx, img in enumerate(images):
        ax = get_fig_ax(axs, row_idx, idx)
        if len(ax.images) == 0 and len(ax.collections) == 0:
            gray = grayscale_transform(img)
            if idx != 0:
                st.session_state.gray_images[filename]["images"].append(gray)
            ax.imshow(gray, cmap="gray")
            ax.axis("off")
            #ax.set_title("Original" if idx == 0 else f"Gray Aug {idx}")
    #Generate the remaining gray images
    if len(all_images) > MAX_UI_AUG_COUNT:
        for img in all_images[MAX_UI_AUG_COUNT:]:
            gray = grayscale_transform(img)
            st.session_state.gray_images[filename]["images"].append(gray)

def create_cluster_image(all_images, axs, row_idx = 0):
    images = all_images if len(all_images) <= MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    for idx, img in enumerate(images):
        draw = None
        ax = get_fig_ax(axs, row_idx, idx)
        if len(ax.images) == 0 and len(ax.collections) == 0:
            width, height = img.size  # Eine Zeile mit der Länge der Daten
            image = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(image)
            tmp_width, tmp_height = 0, 0
            cluster = info['pixel_cluster_map']
            for group in cluster:
                color = constants.get_color(int(group))
                draw.point((tmp_width, tmp_height), fill=color)
                if(tmp_width == width):
                    tmp_height += 1
                    tmp_width = 0
                tmp_width += 1 
            ax.imshow(image)
            ax.axis("off")
    

def create_main_plot(all_images, axs, row_idx = 0):
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]

    for idx, img in enumerate(images):
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
            with st.spinner(f"Process {filename} ..."):
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
        with st.spinner(f"Augmentation of {filename} completed ... Start visualization"):
            if filename not in st.session_state.fig_png:
                ax_counter = sum(1 for opt in option_buttons_ui if opt) + 1
                cols = constants.AUG_COUNT + 1 if constants.AUG_COUNT < MAX_UI_AUG_COUNT else MAX_UI_AUG_COUNT
                fig, axs = plt.subplots(ax_counter, cols, figsize=(15, 6))


                current_row = 0
                # Punktwolke & Augmentierungen generieren
                if show_point_cloud:
                    create_point_cloud([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1

                #Gray scal ebild generieren
                if show_gray_scale:
                    create_gray_images([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1

                if show_cluster:
                    create_cluster_image([info["original"]] + info["aug_images"], axs, current_row)
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
        label="⬇️ Download augmented images as a ZIP file",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip",
    )

    
if st.session_state.image_results and filename in st.session_state.gray_images:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for filename, info in st.session_state.gray_images.items():
            base_name = filename.rsplit('.', 1)[0]
            for i, img in enumerate(info['images']):
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                zipf.writestr(f"{base_name}_gray_scale_{st.session_state.last_aug}_{i+1}.jpg", buf.getvalue())

    download = st.download_button(
        label="⬇️ Download grayscale images as a ZIP file",
        data=zip_buffer.getvalue(),
        file_name="gray_scale_images.zip",
        mime="application/zip",
    )

    


