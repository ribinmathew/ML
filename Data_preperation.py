"This is the first algoritham that we are gonna do without any data cleaning"

"as usual load the data images"
import cv2
import pandas as pd
from os import listdir
import numpy as np
import csv
import  os
from tqdm import tqdm
#from keras.preprocessing.image import ImageDataGenerator, img_to_array


"this data set is a collection of images"
"so we have to convert these images into a array"
train_Data_path = '/home/machine-learning/Desktop/kaggle/Whale/train/'
train_Data_labels_path = "/home/machine-learning/Desktop/kaggle/Whale/whale_kaggle_train (1).csv"

"data images and image labels is loaded"
train_data = os.listdir(train_Data_path)
train_data_labels = pd.read_csv(train_Data_labels_path)
print(train_data_labels.head())
print(len(train_data))
print(len(train_data_labels))
" so  we are gonna create a function which save the image data and the label in same array"
"so our csv file already contain the name of the image and id, but its not in oder. still we can use it"
"Since we are just simply running this model we are directly gonna create one hot  encoding"
"now we are gonna load the image via opencv and convert it into gray and store it as numpy array"

def image_to_numpy(labels):
    image = labels["Image"]
    label = labels["Id"]
    image_array = []
    image_label_arry = []

    for i in tqdm(range(0,len(image))):


        image_label =  label[i]
       # print(image_label)

        image_label = np.array(image_label)
        #print(image_label)
        image_label_arry.append(image_label)


        path_to_image = os.path.join('/home/machine-learning/Desktop/kaggle/Whale/train/',train_data[i])
        #print(path_to_image)

        image = cv2.imread(path_to_image)
        #cv2.imshow("window",image)
        width = 150
        height= 150
        image = cv2.resize(image,(width,height))
        image = cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
        image = np.array(image)
        #print(image)
        #image = np.array(image)
        image_array.append(np.array(image).flatten())

    value = np.asarray(image_array )



image_to_numpy(train_data_labels)
print("done")



"""
print(image_to_numpy(train_data_labels))
"""

