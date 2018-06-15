"""so we are doing the whale dataset again, the main reason i am redoning this is
due to the problems i faced due to my lack of experiance in the field. first thing
the size of the data went so high so that i am unable to process the data. it was 3.4
gb when i converted the images into numpy array and saved it into an csv file"""


"so again a fresh start"
"our dataset is an image dataset and the labels of that image is given in another file/" \
"that file is an .csv file, so lets load that file."

import pandas as pd
from sklearn import preprocessing
import numpy as np
import csv


data_path = pd.read_csv("/home/machine-learning/PycharmProjects/Whale_kaggle/whale_kaggle_train (1).csv")
#print(data_path.columns)
data = data_path["Id"]
catagorical = preprocessing.LabelEncoder()
catagorical_data= catagorical.fit_transform(data_path["Id"])
catagorical_data = catagorical_data.reshape(-1,1)
#catagorical_data = np.array(catagorical_data)
#print(catagorical_data)


"for one hot encoding we need to convert to catogerical encoding, means we have to convert to numerical data"
"Now we have catogerical data, now lets do the onehot encoding"

one_hot = preprocessing.OneHotEncoder()
one_hot_label = one_hot.fit(catagorical_data)

one_hot_labels = one_hot_label.transform(catagorical_data).toarray()
one_hot_encode = [ ]

for i in range(9849):
    label = one_hot_labels[i]

    label = np.reshape(label,(-1,4251))

    one_hot_encode.append(label)


image_data = []
image_data.append([np.array(data_path["Image"]),np.array(data_path["Id"]),np.array(one_hot_encode)])

image_data = np.asarray(image_data)
#print(image_data[0][2][1].shape)
"finally we have dataset with Id, image, and our last hero one hot encode hero"
"now lets save this data"
np.save("data_image_id_label.npy",image_data)

