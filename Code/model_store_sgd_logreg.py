from datetime import datetime
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, StratifiedKFold
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

output_path = sys.argv[1]
training_camera_name = sys.argv[2]
kfold = sys.argv[3]
training_split = sys.argv[4]
pretrained_model = sys.argv[5]
finetune_camera_name = sys.argv[6]
finetune_training_split = sys.argv[7]

if output_path not in os.listdir("./"):
    os.mkdir(f"{output_path}")

cropped_images_path = "model_store_datasets"
resizer_model_filename = 'transformer_model.sav'
training_camera_image_embeddings_file = f'image_embeddings_file_{training_camera_name}-retrained-10x10.npy'
training_camera_image_embeddings_pca_file = f'image_embeddings_pca_file_{training_camera_name}-retrained-10x10.npy'
training_camera_image_labels_file = f'labels_file_{training_camera_name}-retrained-10x10.npy'

if finetune_camera_name == "":
    finetune_camera_name = training_camera_name ## if validation camera is not specified, validate on the same (training) dataset
    print("Validating on same dataset as training")
if finetune_camera_name != training_camera_name:
    print(f"Validating on {finetune_camera_name}")
    finetune_camera_image_embeddings_file = f'image_embeddings_file_{finetune_camera_name}.npy'
    finetune_camera_image_embeddings_pca_file = f'image_embeddings_pca_file_{finetune_camera_name}-0-99.npy'
    finetune_camera_image_labels_file = f'labels_file_{finetune_camera_name}.npy'
    
if kfold == "":
    kfold_splits = 10
else:
    kfold_splits = int(kfold)
    
if training_split != "":
    training_split = float(training_split)
else:
    training_split = 0.8
    
if finetune_training_split != "":
    finetune_training_split = float(finetune_training_split)
else:
    finetune_training_split = 0.2

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
  
##if false negative = classifying blocked as clear, then false positive = clear as blocked. Thus actual positive class = clear, actual negative class = blocked.
#####                   | Actual clear   || Actual blocked |
#####                   ------------------------------------
##### Predicted clear   | True positive  || False positive |
#####                   ------------------------------------
##### Predicted blocked | False negative || True negative  |

## 0 = clear, 1 = blocked

class ModelStoreModel():
    def __init__(self, best_accuracy=0, camera_name="default-camera", output_path="./"):
        self.best_accuracy = best_accuracy
        self.camera_name = camera_name
        self.output_path = output_path
        self.logisticRegressionModel = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SGDClassifier(
                loss='log_loss',
                eta0=0.005, ## initial learning rate
                tol=1e-8,
                n_iter_no_change=50,
                class_weight='balanced', ## weight each class inversely proportionally to their frequency 
                max_iter=140, ## imbalanced
            ))
        ])
        
    def calculateAccuracy(self, testing_labels, predicted_labels):
        self.accuracy = accuracy_score(testing_labels, predicted_labels)
    
    def calculateF1Score(self, testing_labels, predicted_labels):
        self.f1_score = f1_score(testing_labels, predicted_labels)
        
    def calculatePrecision(self, testing_labels, predicted_labels):
        self.precision = precision_score(testing_labels, predicted_labels)
        
    def calculateRecall(self, testing_labels, predicted_labels):
        self.recall = recall_score(testing_labels, predicted_labels)
        
    def calculateROCAUC(self, true_labels, predicted_labels):
        self.roc_auc = roc_auc_score(true_labels, predicted_labels)
            
    def fit(self, training_embeddings, training_labels):
        print("Fitting model")
        self.logisticRegressionModel.fit(
            training_embeddings,
            training_labels
        )
        
    def finetune(self, finetune_embeddings, finetune_labels):
        print("Finetuning model")
        self.logisticRegressionModel.partial_fit(
            finetune_embeddings,
            finetune_labels
        )
        
    def getMetrics(self, testing_labels, predicted_labels, _print=False):
        '''Sets self.accuracy, self.f1-score, self.precision, self.recall, and self.roc_auc)'''
        self.calculateAccuracy(testing_labels, predicted_labels)
        self.calculateF1Score(testing_labels, predicted_labels)
        self.calculatePrecision(testing_labels, predicted_labels)
        self.calculateRecall(testing_labels, predicted_labels)
        self.calculateROCAUC(testing_labels, predicted_labels)
        if _print:
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nROC-AUC Score={:0.3f}".format(
                self.accuracy, self.precision, self.recall, self.f1_score, self.roc_auc)
            print(stats_text)
           
    def make_confusion_matrix(self, testing_labels, group_names=None, categories='auto', count=True, percent=True, cbar=True, xyticks=True,
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
        confusion_matrix = metrics.confusion_matrix(testing_labels, self.predictions)
        
        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
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
            self.getMetrics(testing_labels, self.predictions)
            #Metrics for Binary Confusion Matrices
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nROC-AUC Score={:0.3f}".format(
                self.accuracy, self.precision, self.recall, self.f1_score, self.roc_auc)
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
            file_path = f'{self.output_path}/confusion-matrix-{self.camera_name}-{filename}-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.png'
        else:
            file_path = f'{self.output_path}/confusion-matrix-{self.camera_name}-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.png'
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
            
    def predict(self, testing_embeddings):
        print("Predicting labels for test embeddings")
        self.predictions = self.logisticRegressionModel.predict(testing_embeddings)
        
    def saveModel(self, filename_suffix=""):
        model_file_name = f"{self.output_path}/logistic-regression-model-{self.camera_name}-{filename_suffix}.pkl"
        pickle.dump(self.logisticRegressionModel, open(model_file_name, 'wb'))
        print("Model saved")
        
    def getPredictions(self):
        return self.predictions
    
    def loadModel(self, filename):
        self.logisticRegressionModel = pickle.load(open(filename, 'rb'))
        print("Model loaded")

if resizer_model_filename not in os.listdir("./"):
    feature_extractor_model = DeepResizer().eval()
    print("Saving model...")
    pickle.dump(feature_extractor_model, open(resizer_model_filename, 'wb'))
    print("Model saved")
else:
    print("Retrieving model...")
    feature_extractor_model = pickle.load(open(resizer_model_filename, 'rb'))
    print("Model loaded")
    
transform = transforms.Compose([
    transforms.PILToTensor()
])

if training_camera_image_embeddings_pca_file in os.listdir("./"):
    training_camera_images_embeddings_pca = np.load(training_camera_image_embeddings_pca_file)
    os.remove(training_camera_image_embeddings_pca_file)
else:
    training_camera_images_embeddings_pca = []
if training_camera_image_embeddings_file in os.listdir("./"):
    training_camera_images_embeddings = np.load(training_camera_image_embeddings_file)
    os.remove(training_camera_image_embeddings_file)
else:
    print(f"Can't find {training_camera_image_embeddings_file}")
    training_camera_images_embeddings = []
        
if training_camera_image_labels_file in os.listdir("./"):
    training_camera_images_labels = np.load(training_camera_image_labels_file)
    os.remove(training_camera_image_labels_file)
else:
    print(f"Can't find {training_camera_image_labels_file}")
    training_camera_images_labels = []
    
## if the embeddings or labels are blank, delete everything and start again
if (len(training_camera_images_embeddings) == 0) or (len(training_camera_images_labels) == 0):
    training_camera_images_embeddings = []
    training_camera_images_labels = []
    clear_images = os.listdir(f"{cropped_images_path}/{training_camera_name}/clear")
    blocked_images = os.listdir(f"{cropped_images_path}/{training_camera_name}/blocked")       
    print(f"Starting: {training_camera_name}")
    for clear_image in tqdm(clear_images):
        image = Image.open(f"{cropped_images_path}/{training_camera_name}/clear/{clear_image}")
        image_tensor = transform(image)
        embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
        training_camera_images_embeddings.append(np.array(embedding.detach().numpy()))
        training_camera_images_labels.append(0) ### 0 indicates clear
    for blocked_image in tqdm(blocked_images):
        image = Image.open(f"{cropped_images_path}/{training_camera_name}/blocked/{blocked_image}")
        image_tensor = transform(image)
        embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
        training_camera_images_embeddings.append(np.array(embedding.detach().numpy()))
        training_camera_images_labels.append(1) ### 1 indicates blocked
        
    training_camera_images_embeddings = np.array(training_camera_images_embeddings)
    training_camera_images_labels = np.array(training_camera_images_labels)
        
if finetune_camera_name != training_camera_name:
    if finetune_camera_image_embeddings_pca_file in os.listdir("./"):
        finetune_camera_images_embeddings_pca = np.load(finetune_camera_image_embeddings_pca_file)
        os.remove(finetune_camera_image_embeddings_pca_file)
    else:
        finetune_camera_images_embeddings_pca = []
    if finetune_camera_image_embeddings_file in os.listdir("./"):
        finetune_camera_images_embeddings = np.load(finetune_camera_image_embeddings_file)
        os.remove(finetune_camera_image_embeddings_file)
    else:
        print(f"Can't find {finetune_camera_image_embeddings_file}")
        finetune_camera_images_embeddings = []
            
    if finetune_camera_image_labels_file in os.listdir("./"):
        finetune_camera_images_labels = np.load(finetune_camera_image_labels_file)
        os.remove(finetune_camera_image_labels_file)
    else:
        print(f"Can't find {finetune_camera_image_labels_file}")
        finetune_camera_images_labels = []
        
    ## if the embeddings or labels are blank, delete everything and start again
    if (len(finetune_camera_images_embeddings) == 0) or (len(finetune_camera_images_labels) == 0):
        finetune_camera_images_embeddings = []
        finetune_camera_images_labels = []
        clear_images = os.listdir(f"{cropped_images_path}/{finetune_camera_name}/clear")
        blocked_images = os.listdir(f"{cropped_images_path}/{finetune_camera_name}/blocked")       
        print(f"Starting: {finetune_camera_name}")
        for clear_image in tqdm(clear_images):
            image = Image.open(f"{cropped_images_path}/{finetune_camera_name}/clear/{clear_image}")
            image_tensor = transform(image)
            embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
            finetune_camera_images_embeddings.append(np.array(embedding.detach().numpy()))
            finetune_camera_images_labels.append(0) ### 0 indicates clear
        for blocked_image in tqdm(blocked_images):
            image = Image.open(f"{cropped_images_path}/{finetune_camera_name}/blocked/{blocked_image}")
            image_tensor = transform(image)
            embedding = feature_extractor_model(image_tensor.unsqueeze(0).float())
            finetune_camera_images_embeddings.append(np.array(embedding.detach().numpy()))
            finetune_camera_images_labels.append(1) ### 1 indicates blocked
            
        finetune_camera_images_embeddings = np.array(finetune_camera_images_embeddings)
        finetune_camera_images_labels = np.array(finetune_camera_images_labels)
        
if len(training_camera_images_embeddings_pca) != 0:
    np.save(training_camera_image_embeddings_pca_file, training_camera_images_embeddings_pca)
np.save(training_camera_image_embeddings_file, training_camera_images_embeddings)
np.save(training_camera_image_labels_file, training_camera_images_labels)
if finetune_camera_name != training_camera_name:
    if len(finetune_camera_images_embeddings_pca) != 0:
        np.save(finetune_camera_image_embeddings_pca_file, finetune_camera_images_embeddings_pca)
    np.save(finetune_camera_image_embeddings_file, finetune_camera_images_embeddings)
    np.save(finetune_camera_image_labels_file, finetune_camera_images_labels)
    
## reduce size of dataset to a more manageable size using PCA so we don't run into memory issues
if len(training_camera_images_embeddings_pca) == 0:
    print("Data set contains %d samples with %d features" % training_camera_images_embeddings.shape)
    pca: PCA = PCA(n_components=20)
    training_camera_images_embeddings_pca = pca.fit_transform(training_camera_images_embeddings)
    # Determine the amount of variance explained by these components
    print("Total variance explained = ", np.sum(pca.explained_variance_ratio_))
    print("Saving PCA transformed data")
    np.save(training_camera_image_embeddings_pca_file, np.array(training_camera_images_embeddings_pca))
    
if (finetune_camera_name != training_camera_name) and (len(finetune_camera_images_embeddings_pca) == 0):
    print("Data set contains %d samples with %d features" % finetune_camera_images_embeddings.shape)
    pca: PCA = PCA(n_components=20)
    finetune_camera_images_embeddings_pca = pca.fit_transform(finetune_camera_images_embeddings)
    # Determine the amount of variance explained by these components
    print("Total variance explained = ", np.sum(pca.explained_variance_ratio_))
    print("Saving PCA transformed data")
    np.save(finetune_camera_image_embeddings_pca_file, np.array(finetune_camera_images_embeddings_pca))
    
print("Data set contains %d samples with %d features" % training_camera_images_embeddings_pca.shape)
print("Data set contains %d samples with %d features" % training_camera_images_embeddings.shape)
if finetune_camera_name != training_camera_name:
    print("Data set contains %d samples with %d features" % finetune_camera_images_embeddings.shape)

if (finetune_camera_name == training_camera_name):
    x_train_test, x_validate, y_train_test, y_validate = train_test_split(
                                                                            training_camera_images_embeddings_pca,
                                                                            training_camera_images_labels,
                                                                            train_size=training_split,
                                                                            stratify=training_camera_images_labels
    )
else:
    x_train_test, _, y_train_test, _ = train_test_split(
                                                            training_camera_images_embeddings_pca,
                                                            training_camera_images_labels,
                                                            train_size=training_split,
                                                            stratify=training_camera_images_labels
    )
    if finetune_training_split == 0:
        _, x_validate, _, y_validate = train_test_split(
                                                            finetune_camera_images_embeddings_pca,
                                                            finetune_camera_images_labels,
                                                            train_size=1/len(finetune_camera_images_embeddings) ## can't be 0, so exclude exactly 1 image
        )
    else:
        x_finetune_train, x_validate, y_finetune_train, y_validate = train_test_split(
                                                                                        finetune_camera_images_embeddings_pca,
                                                                                        finetune_camera_images_labels,
                                                                                        train_size=finetune_training_split,
                                                                                        stratify=finetune_camera_images_labels
        )
logReg = ModelStoreModel(camera_name=training_camera_name, output_path=output_path)
if pretrained_model == "":
    stratifiedkfold = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

    k_fold_iterator = 0
    for train_index, test_index in tqdm(stratifiedkfold.split(x_train_test, y_train_test)):
        k_fold_iterator += 1
        x_train_fold, x_test_fold = x_train_test[train_index], x_train_test[test_index]
        y_train_fold, y_test_fold = y_train_test[train_index], y_train_test[test_index]
        print(np.array(x_train_fold).shape, np.array(y_train_fold).shape)
        print(np.array(x_test_fold).shape, np.array(y_test_fold).shape)
        logReg.fit(x_train_fold, y_train_fold)
        logReg.predict(x_test_fold)
        logReg.make_confusion_matrix(testing_labels=y_test_fold,
                                    group_names=['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
                                    categories=['Clear', 'Blocked'], title=training_camera_name, filename=k_fold_iterator)
    ## Save a model trained on all the data
    print("Stratified k-folds complete. Re-fitting model on full training data.")
    logReg.fit(x_train_test, y_train_test)
    logReg.saveModel('trained')
else:
    logReg.loadModel(pretrained_model)
    
if (finetune_camera_name != training_camera_name) and (finetune_training_split != 0):
    logReg.finetune(x_finetune_train, y_finetune_train)
    validation_or_finetuned = " fine tuned on " + finetune_camera_name
elif (finetune_camera_name != training_camera_name) and (finetune_training_split == 0):
    validation_or_finetuned = " predicts " + finetune_camera_name + "(no fine-tuning)"
else:
    validation_or_finetuned = " validated"
### validate against validation set once the accuracies are good enough
logReg.predict(x_validate)
logReg.make_confusion_matrix(testing_labels=y_validate,
                             group_names=['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
                             categories=['Clear', 'Blocked'], title=training_camera_name + validation_or_finetuned, filename='validation')
logReg.getMetrics(y_validate, logReg.getPredictions(), True)