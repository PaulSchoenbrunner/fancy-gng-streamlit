import dbl_gng 
import constants
import clustering
import view
import parser
import color_pca
import numpy as np
from tqdm import trange

# Start fancy DBL GNG

file_list = parser.generate_file_list() # Generate a list of files
data, size_images = parser.parse(file_list) # Parse files and extract image data and sizes
data_names = parser.get_image_names(file_list) # Get image names from the file list

# Initialize lists to store cluster counts and final node counts
cluster_counts = list()
final_nodes_count = list()

epoch = constants.EPOCH # Constant 'EPOCH' from constants.py

# Initialize variables to store final distances and nodes
finalDist = 0
finalNodes = 0

# Loop over all files in the list
for data_index in range(len(file_list)):
   
    
    gng = dbl_gng.DBL_GNG(3, constants.MAX_NODES) # Create a DBL_GNG object with 3 dimensions and MAX_NODES from constants.py
    gng.initializeDistributedNode(data[data_index], constants.STARTING_NODES) # Initialize distributed nodes with current data, note typo: STARTING_NODES

    # Run training epochs
    bar = trange(epoch) # Progress bar for epochs
    for i in bar:
        gng.resetBatch() # Reset batch learning status
        gng.batchLearning(data[data_index]) # Perform batch learning
        gng.updateNetwork() # Update the network
        gng.addNewNode(gng) # Add new nodes
        bar.set_description(f"Epoch: {data_names[data_index]}, Nodes: {len(gng.W)}") # Show epoch and node count in progress bar
    
    gng.cutEdge() # Remove unnecessary connections in the network

    gng.finalNodeDatumMap(data[data_index]) # Create final mapping of nodes to data points
    finalDistMap = gng.finalDistMap # Retrieve final distance mapping
    finalNodes = (gng.W * constants.MAX_COLOR_VALUE).astype(int) # Compute final node values scaled by MAX_COLOR_VALUE
    connectiveMatrix = gng.C # Retrieve node connectivity matrix
    
    # Perform clustering based on final nodes and distances
    pixel_cluster_map, node_cluster_map = clustering.cluster(finalDistMap, finalNodes, connectiveMatrix)
    pixel_cluster_map = np.array(pixel_cluster_map) # Convert cluster mapping to numpy array

    
    cluster_count = int(max(node_cluster_map)) + 1 # Compute number of clusters (max node ID + 1)


    # Save cluster count and final node count
    cluster_counts.append(int(cluster_count))
    final_nodes_count.append(len(finalNodes) + 1)

    # Debugging: visualize network
    # view.view_nodes(finalNodes, node_cluster_map, cluster_count)

    # Perform multiple data augmentations -> TODO: optionally insert/adjust your augmentation code here
    for aug_count in trange(constants.AUG_COUNT):
        # print(f"Augmentation: {str(data_names[data_index])}, {aug_count + 1}/{constants.AUG_COUNT}") # Debugging
        aug_data = color_pca.modify_clusters(
            data[data_index], pixel_cluster_map, cluster_count, size_images, data_index
        ) # Modify cluster colors via PCA
        parser.save_data(
            aug_data, data_names[data_index], size_images, aug_count, data_index, path='./out_data'
        ) # Save augmented data to output folder

        # Debugging: visualize smoothing of nodes and clusters
        # view.show_smoothing(finalNodes, pixel_cluster_map, node_cluster_map, size_images, data_index)
