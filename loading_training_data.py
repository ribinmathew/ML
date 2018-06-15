import cv2
import numpy as np
import csv
data = "/home/machine-learning/PycharmProjects/Whale_kaggle/72.jpg"
image = cv2.imread(data)
image = np.array(image).flatten()
#print(image)
value = np.asarray(image, )
value = value.flatten()
print(value)
with open("img_pixels.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(value)