from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn
import tensorflow as tf
import numpy as np
import tqdm

LR = 0.001

#data = '/home/machine-learning/PycharmProjects/dog_vs_cat/train_data.csv.npy'
Model_name =  'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

convent = input_data(shape=[None,50,50,1],name='input')
convent = conv_2d(convent,32,5,activation='relu')
convent = max_pool_2d(convent,5)

convent = conv_2d(convent,64,5,activation='relu')
convent = max_pool_2d(convent,5)

convent = conv_2d(convent,128,5,activation='relu')
convent = max_pool_2d(convent,5)

convent = conv_2d(convent,64,5,activation='relu')
convent = max_pool_2d(convent,5)

convent = conv_2d(convent,32,5,activation='relu')
convent = max_pool_2d(convent,5)

convent = conv_2d(convent,64,5,activation='relu')
convent = max_pool_2d(convent,5)



convent = fully_connected(convent,1024,activation='relu')
convent = dropout(convent,0.8)

convent = fully_connected(convent,2,activation='softmax')
convent = regression(convent,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='targets')


model = tflearn.DNN(convent,tensorboard_dir = 'log')

import numpy as np



data ='/home/machine-learning/PycharmProjects/dog_vs_cat/train_data.csv.npy'
image_data = np.load(data)
print(image_data)

train = image_data[:-500]
test = image_data[-500:]

train_X = np.array([i[0] for i in train]).reshape(-1,50,50,1)
train_Y = [i[1] for i in train]


test_x = np.array([i[0] for i in test]).reshape(-1,50,50,1)
test_y = [i[1] for i in test]



model.fit({'input': train_X}, {'targets': train_Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=Model_name)
model.save(Model_name)


test_data = '/home/machine-learning/PycharmProjects/dog_vs_cat/test_data.npy'
test_data = np.load(test_data)


with open('submission_file.csv', 'w') as f:
    f.write('id,label\n')

with open('submission_file.csv', 'a') as f:
    for data in test_data:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(50, 50, 1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num, model_out[1]))
