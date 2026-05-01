import csv
from collections import Counter
from datetime import datetime
import math
import os
import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from scipy import spatial
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

images_folder = sys.argv[1]
test_camera = sys.argv[2]
model_store_models = sys.argv[3]
output_path = sys.argv[4]
filename_prefix = sys.argv[5]
finetuning_clear_min = sys.argv[6]
finetuning_clear_max = sys.argv[7]
finetuning_blocked_min = sys.argv[8]
finetuning_blocked_max = sys.argv[9]
blockage_threshold = sys.argv[10]

if finetuning_blocked_min == "":
    finetuning_blocked_min = 0
else:
    finetuning_blocked_min = int(finetuning_blocked_min)
    
if finetuning_clear_min == "":
    finetuning_clear_min = 0
else:
    finetuning_clear_min = int(finetuning_clear_min)

if finetuning_blocked_max == "":
    finetuning_blocked_max = 0
else:
    finetuning_blocked_max = int(finetuning_blocked_max)
    
if finetuning_clear_max == "":
    finetuning_clear_max = 0
else:
    finetuning_clear_max = int(finetuning_clear_max)
    
if finetuning_blocked_min >= finetuning_blocked_max:
    finetuning_blocked_min = 0

if finetuning_clear_min >= finetuning_clear_max:
    finetuning_clear_min = 0
    
if blockage_threshold == "":
    blockage_threshold = 50
else:
    blockage_threshold = int(blockage_threshold)
    
cropped_images_pca_filename = filename_prefix + 'pca_filename.npy'
cropped_images_labels_filename = filename_prefix + 'labels_filename.npy'
cropped_images_filename = filename_prefix + 'filename.npy'
cropped_image_names_filename = filename_prefix + 'names_filename.npy'
model_filename = 'transformer_model.sav'
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

class ModelStore():
    def __init__(self, cBCG, cKCP, dB, sBD, output_path, finetune_parameter_blocked, finetune_parameter_clear): ## accept file paths for the 4 models in the model store
        self.cBCG_location = cBCG
        self.cKCP_location = cKCP
        self.dB_location = dB
        self.sBD_location = sBD
        self.predictions = []
        self.labels = []
        self.output_path = output_path
        self.finetune_parameter_blocked = finetune_parameter_blocked
        self.finetune_parameter_clear = finetune_parameter_clear
        
        self.loadModel('cBCG', self.cBCG_location)
        self.loadModel('cKCP', self.cKCP_location)
        self.loadModel('dB', self.dB_location)
        self.loadModel('sBD', self.sBD_location)
        self.initialiseKNN()
    
    def initialiseKNN(self):
        self.pca_modelpoints_dict: dict = pickle.load(open('pca-modelpoints.npy', 'rb'))
        self.pca_modelpoints = np.array([list(key) for key in self.pca_modelpoints_dict.keys()])
        self.pca_knn = NearestNeighbors(n_neighbors=100).fit(self.pca_modelpoints)
    
    def loadModel(self, filename, filepath):
        if filename == 'cBCG':
            self.cBCG_pipeline: Pipeline = pickle.load(open(filepath, 'rb'))
            self.cBCG_scaler : StandardScaler = self.cBCG_pipeline.named_steps['scaler']
            self.cBCG : SGDClassifier = self.cBCG_pipeline.named_steps['model']
            print(self.cBCG.coef_[0][:5])

        elif filename == 'cKCP':
            self.cKCP_pipeline: Pipeline = pickle.load(open(filepath, 'rb'))
            self.cKCP_scaler : StandardScaler = self.cKCP_pipeline.named_steps['scaler']
            self.cKCP : SGDClassifier = self.cKCP_pipeline.named_steps['model']
            print(self.cKCP.coef_[0][:5])

        elif filename == 'dB':
            self.dB_pipeline: Pipeline = pickle.load(open(filepath, 'rb'))
            self.dB_scaler : StandardScaler = self.dB_pipeline.named_steps['scaler']
            self.dB : SGDClassifier = self.dB_pipeline.named_steps['model']
            print(self.dB.coef_[0][:5])

        elif filename == 'sBD':
            self.sBD_pipeline: Pipeline = pickle.load(open(filepath, 'rb'))
            self.sBD_scaler : StandardScaler = self.sBD_pipeline.named_steps['scaler']
            self.sBD : SGDClassifier = self.sBD_pipeline.named_steps['model']
            print(self.sBD.coef_[0][:5])

        else:
            print("Failed to load model")
            return
        print(f"Model {filename} loaded")
            
    def queryModel(self, model_array, image_embedding):
        model = model_array
        prediction = -1
        if model == 'Cornwall_BudeCedarGrove':
            prediction = self.cBCG_pipeline.predict(image_embedding)
        elif model == 'Cornwall_KingsandCP':
            prediction = self.cKCP_pipeline.predict(image_embedding)
        elif model == 'Devon_Buckfastleigh':
            prediction = self.dB_pipeline.predict(image_embedding)
        elif model == 'sites_brutondam_cam1':
            prediction = self.sBD_pipeline.predict(image_embedding)
        else:
            print(f"Model unknown")
        return prediction
    
    def finetuneModels(self, finetune_image_embeddings, finetune_image_labels):
        finetune_image_embeddings_shuffled, finetune_image_labels_shuffled = shuffle(finetune_image_embeddings, finetune_image_labels, random_state=1)
        self.cBCG_scaler.partial_fit(finetune_image_embeddings_shuffled)
        scaled_embeddings = self.cBCG_scaler.transform(finetune_image_embeddings_shuffled)
        self.cBCG.partial_fit(scaled_embeddings, finetune_image_labels_shuffled)
        
        finetune_image_embeddings_shuffled, finetune_image_labels_shuffled = shuffle(finetune_image_embeddings, finetune_image_labels, random_state=2)
        self.cKCP_scaler.partial_fit(finetune_image_embeddings_shuffled)
        scaled_embeddings = self.cKCP_scaler.transform(finetune_image_embeddings_shuffled)
        self.cKCP.partial_fit(scaled_embeddings, finetune_image_labels_shuffled)
        
        finetune_image_embeddings_shuffled, finetune_image_labels_shuffled = shuffle(finetune_image_embeddings, finetune_image_labels, random_state=3)
        self.dB_scaler.partial_fit(finetune_image_embeddings_shuffled)
        scaled_embeddings = self.dB_scaler.transform(finetune_image_embeddings_shuffled)
        self.dB.partial_fit(scaled_embeddings, finetune_image_labels_shuffled)
        
        finetune_image_embeddings_shuffled, finetune_image_labels_shuffled = shuffle(finetune_image_embeddings, finetune_image_labels, random_state=4)
        self.sBD_scaler.partial_fit(finetune_image_embeddings_shuffled)
        scaled_embeddings = self.sBD_scaler.transform(finetune_image_embeddings_shuffled)
        self.sBD.partial_fit(scaled_embeddings, finetune_image_labels_shuffled)
    
    def calculateAccuracy(self, testing_labels, predicted_labels):
        self.accuracy = accuracy_score(testing_labels, predicted_labels)
    
    def calculateF1Score(self, testing_labels, predicted_labels):
        self.f1_score = f1_score(testing_labels, predicted_labels)
        
    def calculatePrecision(self, testing_labels, predicted_labels):
        self.precision = precision_score(testing_labels, predicted_labels)
        
    def calculateRecall(self, testing_labels, predicted_labels):
        self.recall = recall_score(testing_labels, predicted_labels)
    
    def calculateBalancedAccuracy(self, testing_labels, predicted_labels):
        self.balanced_accuracy = balanced_accuracy_score(testing_labels, predicted_labels)
        
    def calculateROCAUC(self, true_labels, predicted_labels):
        self.roc_auc = roc_auc_score(true_labels, predicted_labels)
    
    def getMetrics(self, _print=False):
        '''Sets self.accuracy, self.f1-score, self.precision, self.recall, self.balanced_accuracy, and self.roc_auc)'''
        self.calculateAccuracy(self.labels, self.predictions)
        self.calculateF1Score(self.labels, self.predictions)
        self.calculatePrecision(self.labels, self.predictions)
        self.calculateRecall(self.labels, self.predictions)
        self.calculateBalancedAccuracy(self.labels, self.predictions)
        self.calculateROCAUC(self.labels, self.predictions)
        if _print:
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nBalanced Accuracy={:0.3f}\nROC-AUC Score={:0.3f}".format(
                self.accuracy, self.precision, self.recall, self.f1_score, self.balanced_accuracy, self.roc_auc)
            print(stats_text)
    
    def pipeline(self, pca_image_embedding):
        '''
        The Querier that accepts an image-embedding, determines image similarity, and queries the relevant model store models for their classifications
        '''        
        ## 1. Compare 20D embedding with the 4 models' 20D PCA-embedded points in order to select the models to query
        # count of how many of these 4 models' points are within the closest 100 neighbours of the image's 20D space.
        # model weighting determined by number of points classified as that model within the 100-neighbour net.
        _, neighbour_indices = self.pca_knn.kneighbors(pca_image_embedding)
        neighbours = list(self.pca_modelpoints[neighbour_indices[0]])
        self.neighbour_models = Counter([tuple(self.pca_modelpoints_dict[tuple(neighbour)]) for neighbour in neighbours])
        
        ## 2. Query the chosen models.
        self.classification_predictions = []
        for model, count in self.neighbour_models.items():
            (model_name, ) = model
            model_name = str(model_name)
            classification = self.queryModel(model_name, np.array(pca_embedding).reshape(1, -1))
            for _ in range(count):
                self.classification_predictions.append(classification)
                    
        ## 3. Average the results. To avoid false negatives, if even split of blocked and clear, return blocked
        # average_prediction = np.average(self.classification_predictions)
        number_predicted_blocked = self.classification_predictions.count(1) # blocked is represented by 1; 0 = clear
        if number_predicted_blocked == 0:
            return 0
        else:
            limit = blockage_threshold # normalised distances means the total weighting is out of 100. So tie-break occurs at 50:50
            if number_predicted_blocked >= limit:
                return 1
            else:
                return 0
        
    def validate(self, image_name, pca_image_embedding, label, total_number = 1):
        prediction = self.pipeline(pca_image_embedding)
        self.predictions.append(prediction)
        self.labels.append(label)
        with open(f"{output_path}/classifications-finetuned-{self.finetune_parameter_blocked}-{self.finetune_parameter_clear}.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([image_name, pca_image_embedding, self.neighbour_models, self.classification_predictions, prediction, label])
        if (len(self.labels) == total_number):
            ## Save the confusion matrix every 1000 images and at the end
            self.make_confusion_matrix(group_names=['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
                                    categories=['Clear', 'Blocked'])
        
    def make_confusion_matrix(self, group_names=None, categories='auto', count=True, percent=True, cbar=True, xyticks=True,
                          xyplotlabels=True, sum_stats=True, figsize=None, cmap='Blues', title=None, filename=None):
        '''
        This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
        Credits to Dennis Trimarchi, https://github.com/DTrimarchi10/confusion_matrix.

        Arguments
        ---------
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        
        count:         If True, show the raw number in the confusion matrix. Default is True.
        
        percent:       If True, show the proportions (percentage) for each category. Default is True.
        
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix. Default is True.
        
        xyticks:       If True, show x and y ticks. Default is True.
        
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        
        sum_stats:     If True, display summary statistics below the figure. Default is True.
        
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                       See http://matplotlib.org/examples/color/colormaps_reference.html
                       
        title:         Title for the heatmap. Default is None.

        '''
        confusion_matrix = metrics.confusion_matrix(self.labels, self.predictions)
        
        ## Generate text inside each square of the confusion matrix
        blanks = ['' for _ in range(confusion_matrix.size)]

        if group_names and len(group_names)==confusion_matrix.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(value) for value in confusion_matrix.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten()/np.sum(confusion_matrix)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(confusion_matrix.shape[0],confusion_matrix.shape[1])

        # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
        if sum_stats:
            self.getMetrics()
            #Metrics for Binary Confusion Matrices
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nBalanced Accuracy={:0.3f}\nROC-AUC Score={:0.3f}".format(
                self.accuracy, self.precision, self.recall, self.f1_score, self.balanced_accuracy, self.roc_auc)
        else:
            stats_text = ""

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize==None:
            #Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks==False:
            #Do not show categories if xyticks is False
            categories=False

        # MAKE THE HEATMAP VISUALIZATION
        plt.figure(figsize=figsize)
        sns.heatmap(confusion_matrix, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)
        
        if title:
            plt.title(title)
            
        if filename != None:
            file_path = f'{self.output_path}/confusion-matrix-tl-querier-{filename}-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-finetuned-{self.finetune_parameter_blocked}-{self.finetune_parameter_clear}.png'
        else:
            file_path = f'{self.output_path}/confusion-matrix-tl-querier-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}-finetuned-{self.finetune_parameter_blocked}-{self.finetune_parameter_clear}.png'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
# 1. Process the image(s) into the DeepResizer feature-embedding format
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

if cropped_images_filename in os.listdir("./"):
    cropped_images = list(np.load(cropped_images_filename))
else:
    cropped_images = []
if cropped_image_names_filename in os.listdir("./"):
    cropped_image_names = list(np.load(cropped_image_names_filename))
else:
    cropped_image_names = []
if cropped_images_labels_filename in os.listdir("./"):
    cropped_images_labels = list(np.load(cropped_images_labels_filename))
else:
    cropped_images_labels = []
    
np.random.seed(1)

if test_camera == 'Cardiff_Tongwynlais_Rowlatt':
    ### Cardiff specific
    images = os.listdir(f"{images_folder}")
    print(f"Pre-processing: {test_camera}")
    for image_name in images:
        if image_name.startswith('high-risk'):
            cropped_images_labels.append(1)
        elif image_name.startswith('low-risk'):
            cropped_images_labels.append(0)
        else:
            continue ## don't process unknown-risk images
        cropped_image_names.append(f"{test_camera}_{image_name}")
        image = Image.open(f"{images_folder}/{image_name}")
        image_tensor = transform(image)
        embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
        cropped_images.append(np.array(embedding.detach().numpy()))
    print(len(cropped_images_labels), len(cropped_image_names))
elif test_camera == 'Cardiff_Tongwynlais_Smith':
    ### Cardiff specific
    images = os.listdir(f"{images_folder}")
    print(f"Pre-processing: {test_camera}")
    for image_name in images:
        if image_name.startswith('blocked'):
            cropped_images_labels.append(1)
        elif image_name.startswith('unblocked'):
            cropped_images_labels.append(0)
        else:
            continue ## don't process unknown-risk images
        cropped_image_names.append(f"{test_camera}_{image_name}")
        image = Image.open(f"{images_folder}/{image_name}")
        image_tensor = transform(image)
        embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
        cropped_images.append(np.array(embedding.detach().numpy()))
    print(len(cropped_images_labels), len(cropped_image_names))
else:
    images = os.listdir(f"{images_folder}/{test_camera}")
    print(f"Pre-processing: {test_camera}")
    for folder in images:
        for image_name in os.listdir(f"{images_folder}/{test_camera}/{folder}"):
            cropped_image_names.append(f"{test_camera}_{folder}_{image_name}")
            image = Image.open(f"{images_folder}/{test_camera}/{folder}/{image_name}")
            image_tensor = transform(image)
            embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
            cropped_images.append(np.array(embedding.detach().numpy()))
            if folder == 'blocked':
                cropped_images_labels.append(1)
            else:
                cropped_images_labels.append(0)
            
            
np.save(cropped_images_filename, np.array(cropped_images))
np.save(cropped_image_names_filename, np.array(cropped_image_names))
np.save(cropped_images_labels_filename, np.array(cropped_images_labels))
print(f"{test_camera} complete")
print("CAMERAS complete")

# reduce size of dataset to a more manageable size using PCA so we don't run into memory issues
if cropped_images_pca_filename not in os.listdir("./"):
    pca: PCA = PCA(n_components=20)
    cropped_clear_images = np.load(cropped_images_filename)
    print("Data set contains %d samples with %d features" % cropped_clear_images.shape)
    cropped_images_pca = pca.fit_transform(cropped_clear_images) 
    print("Data set contains %d samples with %d features" % cropped_images_pca.shape)
    print("Saving PCA transformed data")
    np.save(cropped_images_filename, np.array(cropped_clear_images))
    np.save(cropped_images_pca_filename, np.array(cropped_images_pca))

cropped_images_pca: np.ndarray = np.load(cropped_images_pca_filename)
print("Data set contains %d samples with %d features" % cropped_images_pca.shape)
cropped_images_labels: np.ndarray = np.load(cropped_images_labels_filename)
print("Labels loaded")
cropped_image_names: np.ndarray = np.load(cropped_image_names_filename)
print("Image names loaded")
cropped_images: np.ndarray = np.load(cropped_images_filename)
print("Image embeddings loaded")

# 2. Instantiate the ModelStore class
path_to_cBCG = f'./{model_store_models}/cBCG_30_0.7/logistic-regression-model-Cornwall_BudeCedarGrove-trained.pkl'
path_to_cKCP = f'./{model_store_models}/cKCP_14_0.7/logistic-regression-model-Cornwall_KingsandCP-trained.pkl'
path_to_dB = f'./{model_store_models}/dB_43_0.7/logistic-regression-model-Devon_Buckfastleigh-trained.pkl'
path_to_sBD = f'./{model_store_models}/sBD_8_0.7/logistic-regression-model-sites_brutondam_cam1-trained.pkl'


# 3a. Finetune the models
with open(f"{output_path}/finetuning-classifications.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Blocked finetuning images', 'Clear finetuning images', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy', 'ROC-AUC Score'])
    
finetune_parameter_pairs = [(blocked, clear) for blocked in np.arange(finetuning_blocked_min + 1, finetuning_blocked_max + 2, 10) for clear in np.arange(finetuning_clear_min + 1, finetuning_clear_max + 2, 10)]
finetune_parameter_pairs.insert(0, (0,0))
    
for finetuning_parameter_blocked, finetuning_parameter_clear in tqdm(finetune_parameter_pairs): ## test finetuning with images clear and blocked datasets 
    modelStore = ModelStore(path_to_cBCG, path_to_cKCP, path_to_dB, path_to_sBD, output_path, finetuning_parameter_blocked, finetuning_parameter_clear)
    print(f"Starting finetuning run: {finetuning_parameter_blocked}-{finetuning_parameter_clear} images")
    if (finetuning_parameter_blocked == 0) and (finetuning_parameter_clear == 0):
        x_validate = cropped_images_pca
        y_validate = cropped_images_labels
        image_names = cropped_image_names
        pca_embeddings = cropped_images_pca
    else:
        blocked_indices = np.where(cropped_images_labels == 1)[0] ## unpack the tuple's only element using [0]
        clear_indices = np.where(cropped_images_labels == 0)[0]
        x_finetune = []
        y_finetune = []
        if finetuning_parameter_blocked > 0:
            np.random.shuffle(blocked_indices)
            finetune_blocked_indices = blocked_indices[:finetuning_parameter_blocked]
            validate_blocked_indices = blocked_indices[finetuning_parameter_blocked:]
            x_finetune += list(cropped_images_pca[finetune_blocked_indices])
            y_finetune += list(cropped_images_labels[finetune_blocked_indices])
        else:
            validate_blocked_indices = blocked_indices
        if finetuning_parameter_clear > 0:
            np.random.shuffle(clear_indices)
            finetune_clear_indices = clear_indices[:finetuning_parameter_clear]
            validate_clear_indices = clear_indices[finetuning_parameter_clear:]
            x_finetune += list(cropped_images_pca[finetune_clear_indices])
            y_finetune += list(cropped_images_labels[finetune_clear_indices])
        else:
            validate_clear_indices = clear_indices
            
        x_validate = list(cropped_images_pca[validate_blocked_indices]) + list(cropped_images_pca[validate_clear_indices])
        y_validate = list(cropped_images_labels[validate_blocked_indices]) + list(cropped_images_labels[validate_clear_indices])
        image_names = list(cropped_image_names[validate_blocked_indices]) + list(cropped_image_names[validate_clear_indices])
        pca_embeddings = list(cropped_images_pca[validate_blocked_indices]) + list(cropped_images_pca[validate_clear_indices])

        ## finetune all model store models at once with all the finetuning data
        if len(y_finetune) > 0:
            modelStore.finetuneModels(np.array(x_finetune, dtype=np.float32), y_finetune)
    print("Finetuning complete.")
    # 3b. Call the validate method for each of the non-finetuned images in the camera dataset
    
    with open(f"{output_path}/classifications-finetuned-{finetuning_parameter_blocked}-{finetuning_parameter_clear}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pca_embedding', 'model_distances', 'model_predictions', 'predicted_classification', 'actual_classification'])

    x_validate = np.array(x_validate, dtype=np.float32)
    pca_embeddings = np.array(pca_embeddings, dtype=np.float32)
    total_number_to_validate = len(x_validate)
    print("Total number to validate: ", total_number_to_validate)

    for pca_image_embedding, label, image_name in tqdm(zip(pca_embeddings, y_validate, image_names)):
        pca_embedding = np.array(pca_image_embedding).reshape(1, -1) ## reshape to 2D array for each image
        modelStore.validate(image_name, pca_embedding, label, total_number_to_validate)
    
    print("Validation complete.")
    with open(f"{output_path}/finetuning-classifications.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([finetuning_parameter_blocked, finetuning_parameter_clear, modelStore.accuracy, modelStore.precision, modelStore.recall, modelStore.f1_score, modelStore.balanced_accuracy, modelStore.roc_auc])
    print("Results written to file.")