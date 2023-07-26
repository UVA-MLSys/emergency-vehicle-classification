#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U efficientnet')


# In[2]:


get_ipython().system('pip install numpy==1.21')


# In[3]:


from keras import applications
from keras import callbacks
from keras.models import Sequential


# In[4]:


import tensorflow as tf
tf.config.list_physical_devices('GPU')


# In[5]:


import efficientnet.keras as efn

model = efn.EfficientNetB7(weights='imagenet')


# In[6]:


get_ipython().system('pip install tqdm')


# In[7]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers


# In[8]:


train_dir = "Emergency_Vehicles/train"
test_dir = "Emergency_Vehicles/test"
train_df = pd.read_csv('Emergency_Vehicles/train.csv')
train_df.head()


# In[9]:


from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout

from keras.callbacks import EarlyStopping



# In[10]:


import time
def upload_model():
    eff_net = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    
    #datagen=ImageDataGenerator(rescale=1./255)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      # range (0-180) within which to randomly rotate pictures
        width_shift_range=0.2,  # fraction of total width to randomly translate pictures
        height_shift_range=0.2, # fraction of total height to randomly translate pictures
        shear_range=0.2,        # randomly applying shear transformations
        zoom_range=0.2,         # randomly zooming inside pictures
        horizontal_flip=True,   # randomly flipping half of the images horizontally
        fill_mode='nearest'     # strategy used for filling in newly created pixels
    )

    batch_size=150
    
    train_df.emergency_or_not=train_df.emergency_or_not.astype(str)
    
    train_generator=datagen.flow_from_dataframe(dataframe=train_df[:1150],directory=train_dir,x_col='image_names',
                                            y_col='emergency_or_not',class_mode='binary',batch_size=batch_size,
                                            target_size=(64,64))


    validation_generator=datagen.flow_from_dataframe(dataframe=train_df[1151:],directory=train_dir,x_col='image_names',
                                                    y_col='emergency_or_not',class_mode='binary',batch_size=50,
                                                    target_size=(64,64))
    
    efficient_net = efn.EfficientNetB7(
    weights='imagenet',
    input_shape=(64,64,3),
    include_top=False,
    pooling='max'
)

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units = 120, activation='relu'))
    model.add(Dropout(0.5))  # randomly sets 50% of input units to 0 at each update during training time
    model.add(Dense(units = 120, activation = 'relu'))
    model.add(Dropout(0.5))  # randomly sets 50% of input units to 0 at each update during training time
    model.add(Dense(units = 1, activation='sigmoid'))
    model.summary()
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    start = time.time()

    early_stop = EarlyStopping(monitor='val_loss', patience=5)  # stop training after the validation loss stops improving for 5 epochs
    history = model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=8,
        validation_data=validation_generator,
        validation_steps=7,
        callbacks=[early_stop]  # early stopping
    )

    end = time.time()

    print(f"Training time: {end - start} seconds")
    
    return model


# In[11]:


model = upload_model()


# In[24]:


from tensorflow.keras.preprocessing import image

def classify_image(image_path):
        img = image.load_img(image_path, target_size=(64, 64))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = x / 255.0

        pred = model.predict(x)

        return 1 if pred>0.75 else 0


# In[26]:


print(classify_image("testCar1.png"))
print(classify_image("testCar2.png"))


# In[ ]:




