# Emergency-Vehicle-Classification

**NOTE** if you are simply interested in the emergency vehicle classifcation using images, that is what this repository contains!

Our sibling repository https://github.com/UVA-MLSys/MLCommons-AV contains how LiDAR and point-cloud objects works with our emergency vehicle detection model, feel free to take a look!

## Data Preparation

1) We used the Emergency Vehicles Identification Kaggle Dataset linked here: https://www.kaggle.com/datasets/abhisheksinghblr/emergency-vehicles-identification
please download this dataset prior to attempting to run the codebase.

     -_To note, make sure your dataset is present on the same path, similar to this structure:_

      ![image](https://github.com/UVA-MLSys/emergency_vehicle_classification/assets/123670192/e11d61c2-d7e7-499b-9716-8c319e620568)

      _As you can see, the dataset is stored in emergency-vehicle-classification and not hidden away in any other file. This is because the Emergency_Vehicles file needs to be ***unzipped*** in EV_identification_updated.ipynb_
   
## Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.x
- TensorFlow
- Keras
- EfficientNet
- Numpy
- OpenCV
- Matplotlib
- Pandas
- tqdm

## Installation

_Follow these steps to properly install this application:_



1) The EfficientNet model was used for training, you can install it using:

<pre>
  !pip install -U efficientnet
</pre>
   
2) Also, version 1.21 of numpy is required for compiling, you can install it using:

<pre>
  !pip install numpy==1.21
</pre>
     
  _Both of these installations are present in the code, but it's just a reminder if changes are made to the code or errors are occurring_

3) The Kernel which was used was TensorFlow 2.10.0/Keras Py3.9

    ![image](https://github.com/UVA-MLSys/emergency_vehicle_classification/assets/123670192/427f3423-3868-4685-b0cd-ffb5fd029ded)



## Training & Testing

_Training & Testing should be performed on a server with dedicated GPUs, it is not recommended to run on your own device as that will take a considerate amount of time!_

*Training and Testing was done through UVA Rivanna Servers*

1) The training data comes from the train file in Emergency-Vehicles, containing 1150 images for training
2) The EfficientNet-B7 model pre-trained on ImageNet is loaded using the efficientnet.keras library.
3) Data augmentation is applied to the training dataset to improve model generalization.
4) The top layers of the EfficientNet-B7 model are replaced with custom layers for binary classification.
5) The model is compiled using the Adam optimizer with a learning rate of 0.0001 and binary cross-entropy loss.
6) Training is performed with early stopping to prevent overfitting.
7) To train the model, execute the code provided in the notebook. The trained model will be saved as "custom_model.keras".

_The **results** from the training & testing (graphs):_

![image](https://github.com/UVA-MLSys/emergency_vehicle_classification/assets/123670192/7d6455df-2133-44ea-aef6-fda3c89cd594)

The results (predictions vs accuracy) of the EV_identification_updated.ipynb fo;e can be observered in the submission_effnet.csv file which is created after the program completely runs, and results can be compared with the **half** accurately labeled test(accurate).csv file if predictions match the actual.

Example:

![image](https://github.com/UVA-MLSys/emergency_vehicle_classification/assets/123670192/8c6a183c-fbc7-45d6-817b-c2bbab662d3a)

## Model Deployment

For the EV_research.ipynb file, which is connected to the LiDAR and point-cloud objects repository, the model is saved at the end of the simulation. The model is then uploaded and used to classify the cropped images of the point-cloud detection algorithm.

You can also deploy the trained model to classify images of emergency vehicles. To classify an image, use the classify_image function provided in the EV_research notebook. Pass the path to the image you want to classify, and it will return 1 for emergency vehicles and 0 for non-emergency vehicles

The way of saving the model in EV_Research can also be done in the EV_identification_updated notebook, using the model.save() function.

## Authors

Farhan Khan, Max Titov, Tanush Siotia, Xin Sun

## Issues

Any issues can be commented in the issues section of Github (https://github.com/UVA-MLSys/emergency_vehicle_classification/issues)
