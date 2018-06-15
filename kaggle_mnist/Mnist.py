import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf

train_data_path = "/home/machine-learning/Desktop/kaggle/mnsit/train.csv"
test_data_path = "/home/machine-learning/Desktop/kaggle/mnsit/test.csv"

train_data1 = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
train_data = train_data1.drop(columns='label',index=1 )
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


print(train_data.shape)
print(train_data.head())
image_size = t

x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





rain_data.shape[1]
print("image size = {0}".format(image_size))
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print("image_width = {0} \nimage_height = {1}".format(image_width,image_height))

label = train_data1["label"]
depth = 10
label1 = tf.one_hot(label,depth)

#print(label)
train_data.insert(0,label1,label1)

print(train_data.head())
