# AI-Spy-With-My-Litter-Eye
University Final Year Project- final code and data

## Visualising the t-SNE and silhouette score plots of the images
1. Download the contents of both the _Code_ and _Trained models_ directories. Ensure these are stored within the same local directory on your machine.
2. Crop the images (Section 1.1)
3. Run the command using a command-line interface
`python3 cluster_all_clear_optimised.py {path-to-image-dataset} {path-to-output-location}`

### Important
This code **has not** been optimised for GPUs. Therefore, if using a large dataset such as the training dataset used in this project (Vandaele, 2023), a computer with sufficient available RAM will be required, and use of a server is recommended to reduce processing time.

## Classification pipeline for a new image
1. Download the `Code` folder and the `Trained models`. Ensure these are stored within the same directory.
2. Crop the images (Section 1.1)
3. Run the model store classification pipeline with fine-tuning gridsearch (Section 1.2)

## 1.1 Cropping images
1. Identify a folder of images whose bounding-box is known, or download the training dataset used in this project from https://doi.org/10.17864/1947.000498 (Vandaele, 2023). If it’s a custom folder, ensure the bounding-box coordinates are added to the dictionary coord_dict.
2. Run the command using a command-line interface
`python3 crop_images.py {path-to-image-dataset} {path-to-output-location}`
3. The cropped images will be stored in your defined output location, in `blockagedetection_dataset` within `all_cropped_images.zip`.

## 1.2 Model store classification pipeline
1. Run the command using a command-line interface
`python3 tl-model-store-finetuning-gridsearch.py {path-to-cropped-image-dataset} {name-of-camera-site} {output-folder} {custom-file-prefix-for-cropped-images} {minimum-number-of-finetuning-images-clear or ""} {maximum-number-of-finetuning-images-clear or ""} {minimum-number-of-finetuning-images-blocked or ""} {maximum-number-of-finetuning-images-clear or ""} {blockage-threshold or ""}` // note, blockage-threshold assumed 50%
2. The cropped images will be stored in your defined output location, in `blockagedetection_dataset` within `all_cropped_images.zip`.
