import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd



data = pd.read_csv("Whale_image_label")
Images= data["Id"]
Images = np.asarray(Images)
Image = np.asarray(Images[1])
Image = Image.reshape((150,150))
print(Image)

data = data["Label"]
data = np.asarray(data)
data = np.reshape(data,(9850,1))
data = pd.DataFrame(data)
data = pd.get_dummies(data)
data = np.asarray(data)


print(data.shape)
print(Images.shape)