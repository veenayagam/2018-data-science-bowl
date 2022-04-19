# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:28:43 2022

@author: Darknman
"""

# Import libraries
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from scipy import io

#%%
#1. Data loading
#1.1. Prepare a list for image and mask
images=[]
masks=[]
file_directory = r'C:\Users\Darknman\Desktop\AI\image_segmentation\2018_data_science_bowl\data\train'

#%%
#1.2. Load the images
image_dir = os.path.join(file_directory,'inputs')
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images.append(img)
    
#1.3. Load the masks
mask_dir = os.path.join(file_directory,'masks')
for mask_file in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks.append(mask)
    
#%%
#1.4. Convert images and masks into numpy array
images_np = np.array(images)
masks_np = np.array(masks)
#%%
#1.5. Check some examples
plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()   

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = masks[i]
    plt.imshow(mask_plot, cmap='gray')
    plt.axis('off')
plt.show()  

#%%
#2. Data preprocessing
#2.1. Expand the mask dimension
masks_np_exp = np.expand_dims(masks_np,axis=-1)
#Check the mask output
print(masks[0].min(),masks[0].max())

#%%
#2.2. Change the mask value (1. normalize the value, 2. encode into numerical encoding)
converted_masks = np.round(masks_np_exp/255)
converted_masks = 1 - converted_masks

#%%
#2.3. Normalize the images
converted_images = images_np / 255.0

#%%
#2.4. Do train-test split
from sklearn.model_selection import train_test_split

SEED=123456
x_train,x_test,y_train,y_test = train_test_split(converted_images,converted_masks,test_size=0.2,random_state=SEED)

#%%
#2.5. Convert the numpy array data into tensor slice
train_x = tf.data.Dataset.from_tensor_slices(x_train)
test_x = tf.data.Dataset.from_tensor_slices(x_test)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
test_y = tf.data.Dataset.from_tensor_slices(y_test)

#%%
#2.6. Zip tensor slice into dataset
train = tf.data.Dataset.zip((train_x,train_y))
test = tf.data.Dataset.zip((test_x,test_y))

#%%
#2.7. Convert into prefetch dataset
BATCH_SIZE = 25
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = len(train_x)//BATCH_SIZE
VALIDATION_STEPS = len(test_x)//BATCH_SIZE
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#%%
#3. Prepare model
# We are going to create a modifed version of U-net
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

# Use the activations of those layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

#Define the upsampling stack
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

#Function to create the entire modified U-net
def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

#Define the model
OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

#%%
#Compile the model and display the model structure
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.summary()
keras.utils.plot_model(model,to_file='2018_data_science_bowl.png')


#%%
#Create a function to display some examples
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
for images, masks in train.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])
    
#%%
#Create a function to process predicted mask
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

#Create a function to display prediction
def show_predictions(dataset=None,num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)[0]])
    else:
        display([sample_image,sample_mask,create_mask(model.predict(sample_image[tf.newaxis,...]))[0]])

#Custom callback to display result during training
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\n Sample prediction after epoch {}\n'.format(epoch+1))

#Tensorboard callback
import datetime
log_dir = r"C:\Users\Darknman\Desktop\AI\Tensorflowboard\2018_data_science_bowl\2018_data_science_bowl_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1,profile_batch=0)


#Start to do training
EPOCH = 10

history = model.fit(train,epochs=EPOCH,steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=test,
                    callbacks=[DisplayCallback(),tb_callback])

#%%
#Deploy model by using the show_prediction functions created before
show_predictions(test,3)
         