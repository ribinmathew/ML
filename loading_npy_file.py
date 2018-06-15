import numpy as np
from os import listdir
import os
import cv2
import time
import tqdm
load_data = np.load("data_image_id_label.npy")
#rint(load_data[0][2][8549])
"so we have image, label, and onehot encode"
" now have to convert the image file into arryay"
"so  we have tried that way and found out the image_data " \
"when converted into csv file it became 3.4 gb file" \
"so now we have to load data via pipeline method" \
"and i dont know how to do it.." \
"so lets learn it and do it"
" first we have to create a file which contain image_path to the image in the data" \
"we have"



image_dir_path = "/home/machine-learning/Desktop/kaggle/Whale/train/"
image_name = load_data[0][0]

#print(load_data[0][2])

image_paths= []
for i in image_name:
    image_path = os.path.join(image_dir_path,i)
    image_paths.append(np.array(image_path))

"""



#image = str(image_paths[0])
#images =  cv2.imread(image)
#cv2.imshow("image",images)
#cv2.waitKey(2000)
#cv2.destroyWindow('image')
#print(images)
#training_images = np.array([ImportImage(img) in image_path])


" so we are ready with label, image_path, and the onehot encoded" \
"now we have to convert the image into grayscale, resize, then flatten it " \
"and keep ready to push into the model"

"""
def image_array(image_path):
    image_array = []

    for i in image_path:
        print(i)
        image = cv2.imread(str(i))
        image = cv2.resize(image,(100,100))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image_array.append(np.array(image).flatten().reshape(-1,100,100))

    return image_array


x_train= np.asarray(image_array(image_paths))
#x_train = x_train.reshape()
y_train = np.asarray(load_data[0][2])

input_shape = x_train[0].shape
print(y_train[0])
print(x_train[0])


"so we have reached a stage where we have labels which is onehot encoded and " \
"image data which is done preprocessing" \
"now lets define a model "

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D,MaxPooling2D
from keras import backend as kd
from keras.preprocessing.image import  ImageDataGenerator



model = Sequential
model.add(Conv2D(48, 3,3,activation='relu',input_shape=(1,100,100)))
model.add(Conv2D(48,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48,5,5,activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(.33))
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model = model.add(Dropout(0.33))
model = model.add(Dense(36,activation='relu'))
model = model.add(Dense(4251,activation='softmax'))

model = model.compile(loss= keras.losses.categorical_crossentropy(),optimizer=keras.optimizers.Adam,metrics=['accuracy'])
model.fit(x_train,y_train,batch_size = 128, epochs = 10, validation_split=0.2)
print(model.summary())

