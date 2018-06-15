import pandas as pd
import keras
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout


train_data_path = "/home/machine-learning/Desktop/kaggle/mnsit/train.csv"
test_data_path = "/home/machine-learning/Desktop/kaggle/mnsit/test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

test_data = test_data.astype('float32')
test_data/=255
test_data= test_data.values.reshape(-1,28,28,1)


def data_prep(row):
    labels = keras.utils.to_categorical(row.label, num_classes=10)

    image_number =row.shape[0]
    images_array = row.values[:,1:]
    reshaped_images = images_array.reshape(image_number, 28,28,1)
    out_images = reshaped_images/255

    return out_images, labels

x,y = data_prep(train_data)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(20,kernel_size=(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam', metrics=['accuracy'])

import numpy as np
model.fit(x,y, batch_size=128,epochs=2, validation_split= 0.2)
res  = model.predict(test_data)
res = np.argmax(res,axis=1)
res_df = pd.DataFrame(data={'ImageId':range(1,28001),'Labels':res})
res_df.to_csv('res.csv',index=False)
model.summary()