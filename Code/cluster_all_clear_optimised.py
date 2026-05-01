import hdbscan
from matplotlib import pyplot as plt
import numpy as np
import os, sys
import openTSNE
import pickle
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

camera_classes = {
    "Cornwall_BodminFlaxmoor": {'color':"black", 'marker':'o'}, # 909 clear images
    "Cornwall_BodminPetrocsWell_Screen": {'color':"dimgray", 'marker':'*'}, # 1188 clear images
    "Cornwall_BoscastleMarine": {'color':"silver", 'marker':'X'}, # 433 clear images
    "Cornwall_BudeCedarGrove": {'color':"lightgrey", 'marker':'p'}, # 855 clear images
    "Cornwall_CawsandCP": {'color':"rosybrown", 'marker':'^'}, # 1434 clear images
    "Cornwall_Crinnis": {'color':"lightcoral", 'marker':'s'}, # 762 clear images
    "Cornwall_GorranHaven": {'color':"indianred", 'marker':'o'}, # 1358 clear images
    "Cornwall_IdlessDam_cam1": {'color':"brown", 'marker':'*'}, # 929 clear images
    "Cornwall_KingsandCP": {'color':"maroon", 'marker':'X'}, # 380 clear images
    "Cornwall_KingsandPP": {'color':"red", 'marker':'p'}, # 1231 clear images
    "Cornwall_LostwithielUP_Screen": {'color':"tomato", 'marker':'^'}, # 1303 clear images
    "Cornwall_LostwithielWW": {'color':"darksalmon", 'marker':'s'}, # 1240 clear images
    "Cornwall_Mevagissey_PreScreen": {'color':"lightsalmon", 'marker':'o'}, # 657 clear images
    "Cornwall_PenrynTP": {'color':"sienna", 'marker':'*'}, # 745 clear images
    "Cornwall_PenzanceCC": {'color':"chocolate", 'marker':'X'}, # 542 clear images
    "Cornwall_PenzanceCS": {'color':"sandybrown", 'marker':'p'}, # 716 clear images
    "Cornwall_PlymptonChaddlewood_MainScreen": {'color':"darkorange", 'marker':'^'}, # 995 clear images
    "Cornwall_PlymptonForSt": {'color':"tan", 'marker':'s'}, # 1203 clear images
    "Cornwall_PlymptonKay": {'color':"wheat", 'marker':'o'}, # 1596 clear images
    "Cornwall_PolperroLangreek": {'color':"darkgoldenrod", 'marker':'*'}, # 1393 clear images
    "Cornwall_PorthlevenScreen": {'color':"gold", 'marker':'X'}, # 204 clear images
    "Cornwall_Portreath": {'color':"khaki", 'marker':'p'}, # 798 clear images
    "Cornwall_StIvesConsols_Screen": {'color':"darkkhaki", 'marker':'^'}, # 1558 clear images
    "Cornwall_StIvesConsols_Upstream": {'color':"olive", 'marker':'s'}, # 1194 clear images
    "Cornwall_TamertonFoliot": {'color':"yellow", 'marker':'o'}, # 231 clear images
    "Cornwall_WadebridgePolmorla": {'color':"olivedrab", 'marker':'*'}, # 132 clear images
    "Devon_BarnstapleBradiford": {'color':"yellowgreen", 'marker':'X'}, # 669 clear images
    "Devon_BarnstapleConeyGut_Screen": {'color':"darkolivegreen", 'marker':'p'}, # 795 clear images
    "Devon_BarnstaplePortmarshLane": {'color':"greenyellow", 'marker':'^'}, # 150 clear images
    "Devon_Buckfastleigh": {'color':"lawngreen", 'marker':'s'}, # 1732 clear images
    "Devon_Dulverton": {'color':"darkseagreen", 'marker':'o'}, # 501 clear images
    "Devon_Holbeam_Impound": {'color':"palegreen", 'marker':'*'}, # 950 clear images
    "Devon_KenwithValleyChannelScreen": {'color':"forestgreen", 'marker':'X'}, # 1035 clear images
    "Devon_LympstoneScreen": {'color':"darkgreen", 'marker':'p'}, # 751 clear images
    "Devon_OtterySMChapelLane": {'color':"mediumaquamarine", 'marker':'^'}, # 1510 clear images
    "Devon_PalmersDam_Upstream": {'color':"lightseagreen", 'marker':'s'}, # 459 clear images
    "Devon_SwimbridgeScreen": {'color':"teal", 'marker':'o'}, # 593 clear images
    "sites_ashtoncollitersbrooksplitter_cam2": {'color':"aqua", 'marker':'*'}, # 73 clear images
    "sites_brutondam_cam1": {'color':"powderblue", 'marker':'X'}, # 219 clear images
    "sites_corsham_cam1": {'color':"deepskyblue", 'marker':'p'}, # 419 clear images
    "sites_corshamaqueduct_cam1": {'color':"steelblue", 'marker':'^'}, # 733 clear images
    "sites_eastville_cam2": {'color':"cornflowerblue", 'marker':'s'}, # 146 clear images
    "sites_midsomernorton_cam1": {'color':"midnightblue", 'marker':'o'}, # 1064 clear images
    "sites_minehead_cam1": {'color':"blue", 'marker':'*'}, # 62 clear images
    "sites_mineheadgate_cam1": {'color':"slateblue", 'marker':'X'}, # 386 clear images
    "sites_motcombe_cam1": {'color':"mediumpurple", 'marker':'p'}, # 1169 clear images
    "sites_northdrain_cam1": {'color':"rebeccapurple", 'marker':'^'}, # 1180 clear images
    "sites_pilloutfall_cam1": {'color':"indigo", 'marker':'s'}, # 344 clear images
    "sites_sheptonmallet_cam1": {'color':"plum", 'marker':'o'}, # 422 clear images
    "sites_sheptonmallet_cam2": {'color':"darkmagenta", 'marker':'*'}, # 1568 clear images
    "sites_sistontunnel_cam1": {'color':"fuchsia", 'marker':'X'}, # 782 clear images
    "sites_southperrott_cam1": {'color':"mediumvioletred", 'marker':'p'}, # 366 clear images
    "sites_stannesscreen_cam1": {'color':"palevioletred", 'marker':'^'}, # 599 clear images
    "sites_uphill_cam1": {'color':"lightpink", 'marker':'s'}, # 140 clear images
}

## Total number of clear images across all 54 classes: 42803
## Total number of images across all 54 classes (blocked and clear): 54,394

model_filename = 'transformer_model.sav'
cropped_clear_images_filename = 'cropped_clear_images_filename-10x10.npy'
cropped_clear_image_names_filename = 'cropped_clear_image_names_filename-10x10.npy'
cropped_clear_images_labels_filename = 'cropped_clear_images_labels_filename-10x10.npy'
cropped_clear_images_pca_filename = 'cropped_clear_images_pca_filename-10x10.npy'
model_store_models = ['Cornwall_BudeCedarGrove', 'Cornwall_KingsandCP', 'Devon_Buckfastleigh', 'sites_brutondam_cam1']

class DeepResizer(nn.Module):
    # Load a pre-trained ResNet-50 backbone
    backbone = models.resnet50(weights='DEFAULT')
    def __init__(self):
        print("Building ResNet50 model")
        super(DeepResizer, self).__init__()
        # Extract layers except the final pooling and FC layers
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Adaptive Pooling ensures a uniform output regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(10) ## m x m square

    def forward(self, x):
        # x can have variable dimensions [Batch, 3, H, W]
        x = self.feature_extractor(x)
        x = self.adaptive_pool(x)
        return torch.flatten(x)
    
if model_filename not in os.listdir("./"):
    feature_extractor_model = DeepResizer().eval()
    print("Saving model...")
    pickle.dump(feature_extractor_model, open(model_filename, 'wb'))
    print("Model saved")
else:
    print("Retrieving model...")
    feature_extractor_model = pickle.load(open(model_filename, 'rb'))
    print("Model loaded")

transform = transforms.Compose([
    transforms.PILToTensor()
])

cropped_clear_images_path = sys.argv[1]
output_path = sys.argv[2]

if output_path not in os.listdir("./"):
    os.mkdir(f"{output_path}")

filename_identifier = "entire-clear-dataset"
plt.rcParams['legend.numpoints'] = 1 ## required for handles to display properly
plt.ioff() ## disable interactive plotting for speed gains
np.random.seed(0)

####### Uncomment the following code the first time you run this script ONLY ###################
## Following code saves embeddings for efficient processing
# for camera in tqdm(os.listdir(f"{cropped_clear_images_path}")):
#     if cropped_clear_images_filename in os.listdir("./"):
#         cropped_clear_images = list(np.load(cropped_clear_images_filename))
#         os.remove(cropped_clear_images_filename)
#     else:
#         cropped_clear_images = []
#     if cropped_clear_image_names_filename in os.listdir("./"):
#         cropped_clear_image_names = list(np.load(cropped_clear_image_names_filename))
#         os.remove(cropped_clear_image_names_filename)
#     else:
#         cropped_clear_image_names = []
#     if cropped_clear_images_labels_filename in os.listdir("./"):
#         cropped_clear_images_labels = list(np.load(cropped_clear_images_labels_filename))
#         os.remove(cropped_clear_images_labels_filename)
#     else:
#         cropped_clear_images_labels = []
        
#     clear_images = os.listdir(f"{cropped_clear_images_path}/{camera}/clear")
#     print(f"Starting: {camera}")
#     for clear_image in tqdm(clear_images):
#         image = Image.open(f"{cropped_clear_images_path}/{camera}/clear/{clear_image}")
#         cropped_clear_image_names.append(clear_image)
#         image_tensor = transform(image)
#         embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
#         cropped_clear_images.append(np.array(embedding.detach().numpy()))
#         cropped_clear_images_labels.append(camera)
        
#     np.save(cropped_clear_images_filename, np.array(cropped_clear_images))
#     np.save(cropped_clear_image_names_filename, np.array(cropped_clear_image_names))
#     np.save(cropped_clear_images_labels_filename, np.array(cropped_clear_images_labels))
#     print(f"{camera} complete")
        
# print("CAMERAS complete")

## reduce size of dataset to a more manageable size using PCA so we don't run into memory issues
# if cropped_clear_images_pca_filename not in os.listdir("./"):
#     pca = PCA(n_components=20)
#     cropped_clear_images = np.load(cropped_clear_images_filename)
#     print("Data set contains %d samples with %d features" % cropped_clear_images.shape)
#     cropped_clear_images_pca = pca.fit_transform(cropped_clear_images)
#     print("Saving PCA transformed data")
#     np.save(cropped_clear_images_filename, np.array(cropped_clear_images))
#     np.save(cropped_clear_images_pca_filename, np.array(cropped_clear_images_pca))
    
cropped_clear_images_pca = np.load(cropped_clear_images_pca_filename)
print("Data set contains %d samples with %d features" % cropped_clear_images_pca.shape)

number_of_cropped_clear_images = len(cropped_clear_images_pca)
cropped_clear_images_labels = np.load(cropped_clear_images_labels_filename)
labels = camera_classes.keys()

model_store_pcadatapoints_dict = {}
print("Storing PCA embeddings in lookup-table...")
for i in tqdm(range(number_of_cropped_clear_images)):
    modelname = cropped_clear_images_labels[i]
    if modelname in model_store_models:
        pca_representation = tuple(cropped_clear_images_pca[i])
        if pca_representation in model_store_pcadatapoints_dict:
            model_store_pcadatapoints_dict[pca_representation].append(modelname)
        else:
            model_store_pcadatapoints_dict[pca_representation] = [modelname]
pickle.dump(model_store_pcadatapoints_dict, open('pca-20-10x10-modelpoints.npy', 'wb'))
    
## following code based on OpenTSNE's documentation for improving TSNE's global structure preservation by using
## - PCA for local structure preservation
## - Cosine distance metric instead of Euclidean as 'the notion of Euclidean distance breaks down in high dimensions'
## https://opentsne.readthedocs.io/en/latest/examples/03_preserving_global_structure/03_preserving_global_structure.html
## https://opentsne.readthedocs.io/en/stable/examples/04_large_data_sets/04_large_data_sets.html

print("Precompute TSNE Affinities...")

aff50 = openTSNE.affinity.PerplexityBasedNN(
    cropped_clear_images_pca,
    perplexity = 50,
    n_jobs = 24,
    random_state = 3,
    verbose = True,
)
print("\taff50 computed")

# Use downsampling for PCA initialisation to reduce memory and CPU usage in large dataset embeddings
# see https://opentsne.readthedocs.io/en/stable/examples/04_large_data_sets/04_large_data_sets.html

indices = np.random.permutation(list(range(cropped_clear_images_pca.shape[0])))
image_permutation_indices = np.argsort(indices)

images_sample, images_rest = cropped_clear_images_pca[indices[:800]], cropped_clear_images_pca[indices[800:]]
np.save(cropped_clear_images_pca_filename, cropped_clear_images_pca)
print("\tSaved cropped_clear_images_pca")

cropped_clear_images_labels = np.load(cropped_clear_images_labels_filename)
labels_sample, labels_rest = cropped_clear_images_labels[indices[:800]], cropped_clear_images_labels[indices[800:]]
np.save(cropped_clear_images_labels_filename, cropped_clear_images_labels)
print("\tSaved cropped_clear_images_labels")

sample_affinities = openTSNE.affinity.PerplexityBasedNN(
    images_sample,
    perplexity = 500,
    n_jobs = 24,
    random_state = 50,
    verbose = True,
)

print("All affinities precomputed")

print("Initialising PCA")
sample_init = openTSNE.initialization.pca(images_sample, random_state=42, verbose=True)
print("Fitting TSNE...")
sample_embedding = openTSNE.TSNE(
    n_jobs = 24,
    verbose = True
).fit(
    affinities=sample_affinities,
    initialization=sample_init,
)
print("TSNE fitting with PCA initialisation - COMPLETE")

rest_init = sample_embedding.prepare_partial(
    images_rest,
    k = 1,
    perplexity = 1/3,
)

full_initialisation = np.vstack((sample_embedding, rest_init))[image_permutation_indices]
scaler = (np.std(full_initialisation[:,0]) * 10000)
pickle.dump(scaler, open('tsne_scaler.npy', 'wb'))
full_initialisation = full_initialisation / scaler

embedding = openTSNE.TSNEEmbedding(
    full_initialisation,
    aff50,
    n_jobs = 24,
    verbose = True,
    random_state = 50,
)

embedding1 = embedding.optimize(n_iter = 500, exaggeration = 10) # early-exaggeration, with alpha*h ~ n/10 from https://doi.org/10.1137/18M1216134
embedding2 = embedding1.optimize(n_iter = 250, exaggeration = 2)
embedding3 = embedding2.optimize(n_iter = 250, exaggeration = 2)
embedding4 = embedding3.optimize(n_iter = 250, exaggeration = 2)
final_embedding = embedding4.optimize(n_iter = 250, exaggeration = 5) # late-exaggeration https://doi.org/10.1038/s41592-018-0308-4
# pickle.dump(final_embedding, open('tsne_optimised-all-clear.npy', 'wb'))

## The silhouette_score gives the average value for all the samples.
## This gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(final_embedding, cropped_clear_images_labels)
print("For n_clusters = 54, the average silhouette_score is : ", silhouette_avg)

legend_handles = []
previous_label = ""

number_of_cropped_clear_images = len(cropped_clear_images_pca)
labels = camera_classes.keys()

fig, ax1 = plt.subplots(1,1)
fig.set_size_inches(18, 35)

# The subplot is the silhouette plot
# The silhouette coefficient can range from -1, 1
ax1.set_xlim([-1, 1])
# The (n_clusters+1)*10 is for inserting blank space between silhouette
# plots of individual clusters, to demarcate them clearly.
ax1.set_ylim([0, len(final_embedding) + (54 + 1) * 10]) ## 54 = number of clusters

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(final_embedding, cropped_clear_images_labels)

fixed_cluster_height = 800
y_lower = 10

silhouette_score_dict = {}
for label in labels:
    # Aggregate the silhouette scores for samples belonging to the same cluster, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cropped_clear_images_labels == label]
    ith_cluster_silhouette_values.sort() ## sorts in place
    
    silhouette_score_dict[label] = np.array(ith_cluster_silhouette_values)

    y_upper = y_lower + fixed_cluster_height
    y_range = np.linspace(y_lower, y_upper, len(ith_cluster_silhouette_values))
    
    len_cluster = len(ith_cluster_silhouette_values)
    
    ax1.fill_betweenx(
        y_range,
        0,
        ith_cluster_silhouette_values,
        facecolor=camera_classes[label]['color'],
        edgecolor=camera_classes[label]['color'],
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster names at the middle
    ax1.text(-0.05, y_lower + 0.5 * fixed_cluster_height, label, fontsize=8)

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.axvline(x=0.85, color="black", linestyle="-")


ax1.set_yticks([])
ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle(
    f"n_clusters = 54, silhouette average = {silhouette_avg}",
    fontsize=14,
    fontweight="bold",
)

plt.savefig(f'{output_path}/silhouettes-{filename_identifier}-pca-20-10x10.png', bbox_inches='tight')
plt.close()

model_store_tsne_centroids_dict = {}
model_store_tsneclusterpoints_dict = {}
model_store_originaldatapoints_dict = {}
for model in model_store_models:
    model_store_tsneclusterpoints_dict[model] = []

print("Scattering data...")
for i in tqdm(range(number_of_cropped_clear_images)):
    modelname = cropped_clear_images_labels[i]
    if modelname in model_store_models:
        tsne_representation = tuple(final_embedding[i])
        if tsne_representation in model_store_tsneclusterpoints_dict:
            model_store_tsneclusterpoints_dict[tsne_representation].append(modelname)
        else:
            model_store_tsneclusterpoints_dict[tsne_representation] = [modelname]
    plt.scatter(*final_embedding[i].T, color=camera_classes[modelname]['color'], marker=camera_classes[modelname]['marker'])
    if modelname != previous_label:
        handle, = plt.plot([], marker=camera_classes[modelname]['marker'], color=camera_classes[modelname]['color'], label=modelname)
        legend_handles.append(handle)
        previous_label = modelname
colours = [camera_classes[label]['color'] for label in labels]
plt.legend(handles=legend_handles, ncol=3, bbox_to_anchor=(1, 1))
plt.title("Multiscale t-SNE with PCA initialisation")
plt.savefig(f'{output_path}/colour-coded-clusters-{filename_identifier}-pca-20-10x10.png', bbox_inches='tight')
plt.close()

print("\n IMAGES saved")