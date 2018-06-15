"first we are going to load the images using pandas,cv"


import cv2
import os
import numpy as np
from tqdm import tqdm

train_data_path = "/home/machine-learning/Desktop/kaggle/dog_vs_cat/train/"
test_data_path  ="/home/machine-learning/Desktop/kaggle/dog_vs_cat/test/"


train_data = os.listdir(train_data_path)
test_data = os.listdir(test_data_path)

""" so we have loaded the data for train and test
Now we have to create labels for the image
Here we are going to use the tensorflow to find the prediction
Since we are using the tensorflow we have to convert these images 
into tensor acceptable format, So we have to do some preprocessing before
we push the images into the tensorflow """
" Preprocessing of image"

"first i am trying to load the image into an array  using numpy"

"here we have defined a function image label, which labels the image [0,1] for cat and [1,0] for dog"

def image_label(image):

    label = image[:3]
    if label =='cat':
        label =[0,1]
        return label
    if label =='dog':
        label = [1,0]
        return label
"Here we have defined another function which will process the image data and " \
" convert that data into array using numpy, here we are saving the image as grayscale" \
"also we are resizing the image into 50X50 square for uniformity, at last the " \
"function will save the processed data in dir..."
def train_data_image():
    array_train_images = []
    for i in tqdm(train_data):
        label = image_label(i)
        image_path = os.path.join(train_data_path,i)
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image =cv2.resize(image,(50,50))
        array_train_images.append([np.array(image),np.array(label)])


    np.save('train_data',array_train_images)
    return array_train_images

def test_data_image():
    array_train_images = []
    for i in tqdm(test_data):
        #label = image_label(i)
        img_num = i.split('.')[0]
        image_path = os.path.join(test_data_path,i)
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image =cv2.resize(image,(50,50))
        array_train_images.append([np.array(image), img_num])


    np.save('test_data',array_train_images)
    return array_train_images

test_data_image()