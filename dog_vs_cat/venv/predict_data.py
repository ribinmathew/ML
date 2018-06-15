import numpy as np
import tqdm
test_data = '/home/machine-learning/PycharmProjects/dog_vs_cat/test_data.npy'

test_data = np.load(test_data)


for  data in tqdm(test_data):
    print(data[1])

#print(test_data)