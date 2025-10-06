import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image
import matplotlib.pyplot as plt
import io, zipfile


#st.set_page_config(page_title="DBL-GNG Image Augmentation", layout="wide")

# --- Session-States initialisieren ---
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "image_results" not in st.session_state:
    st.session_state.image_results = {}  # {filename: {"original": Image, "aug_images": [...], "cluster_count": int, "data_shape": tuple}}
if "fig" not in st.session_state:
    st.session_state.fig = {}
if "fig_png" not in st.session_state:
    st.session_state.fig_png = {}
if "done" not in st.session_state:
    st.session_state.done = False

def reset_session():
    st.session_state.clear()

def reset_for_new_run():
    st.session_state.image_results = {}
    st.session_state.fig = {}
    st.session_state.fig_png = {}
    
# --- Streamlit UI ---
st.title("🧠 DBL-GNG Image Augmentation")
st.write("Lade ein oder mehrere Bilder hoch oder nimm eines mit der Kamera auf.")


# --- Eingabeoption ---
input_option = st.radio("Bildquelle auswählen:", ["Datei-Upload", "Kamera"])

if input_option == "Datei-Upload":
    uploaded_files = st.file_uploader(
        "Bilder auswählen", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        on_change= reset_session
    )
elif input_option == "Kamera":
    uploaded_files = st.camera_input("Bild aufnehmen")

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

start_augmentation = st.button("🚀 Starte Augmentierung")

if start_augmentation and st.session_state.done:
    reset_for_new_run()

def generate_augmentations(image_data, size):
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

def create_plot(all_images):
    fig, axs = plt.subplots(2, len(all_images), figsize=(15, 6))
    for idx, img in enumerate(all_images):
        rgb_image = img.convert("RGB")
        width, height = img.size
        points = np.array([
            (r, g, b, r, g, b)
            for x in range(width)
            for y in range(height)
            for (r, g, b) in [rgb_image.getpixel((x, y))]
        ])
        axs[0, idx].scatter(points[:, 1], points[:, 2], c=points[:, 3:6] / 255, s=1)
        axs[0, idx].set_xlim(0, 255)
        axs[0, idx].set_ylim(0, 255)
        axs[0, idx].set_aspect('equal', 'box')
        axs[0, idx].set_title("Original" if idx == 0 else f"Aug {idx}")
        axs[1, idx].imshow(img)
        axs[1, idx].axis("off")
    fig.tight_layout()
    return fig        

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

# --- Hauptverarbeitung ---
if (start_augmentation or st.session_state.done) and st.session_state.uploaded_files:
    for uploaded_file in st.session_state.uploaded_files:
        filename = uploaded_file.name

        # Falls bereits berechnet, überspringen
        if filename not in st.session_state.image_results:
            with st.spinner(f"Verarbeite {filename} ..."):
                image = Image.open(uploaded_file).convert("RGB")
                image_array = np.asarray(image)
                data_array = image_array.reshape(-1, 3) / constants.MAX_COLOR_VALUE

                aug_images, cluster_count = generate_augmentations(data_array, image.size)
                st.session_state.image_results[filename] = {
                    "original": image,
                    "aug_images": aug_images,
                    "cluster_count": cluster_count,
                    "data_shape": data_array.shape,
                }

        # --- Anzeige ---
        info = st.session_state.image_results[filename]
        st.divider()
        st.subheader(f"📸 {filename}")
        st.write(f"**Bildgröße:** {info['original'].size}")
        st.write(f"**Bildarray-Form:** {info['data_shape']}")
        st.write(f"**Anzahl der Cluster:** {info['cluster_count']}")

        # --- Punktwolke & Augmentierungen anzeigen ---
        if filename not in st.session_state.fig:
            print("New")
            st.session_state.fig[filename] = create_plot([info["original"]] + info["aug_images"])
            png_buf = fig_to_png(st.session_state.fig[filename])
            st.session_state.fig_png[filename] = png_buf.getvalue()
       
        st.image(st.session_state.fig_png[filename])
    st.session_state.done = True
        




# --- Download-Bereich ---
if st.session_state.image_results:
    st.divider()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for filename, info in st.session_state.image_results.items():
            base_name = filename.rsplit('.', 1)[0]
            for i, img in enumerate(info["aug_images"]):
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                zipf.writestr(f"{base_name}_aug_{i+1}.jpg", buf.getvalue())

    st.download_button(
        label="⬇️ Augmentierte Bilder als ZIP herunterladen",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip",
    )


