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



model.fit({'input':train_X},{'targets':train_Y},n_epoch=3,
          validation_set =({'input':test_x},{'target':test_y}),
          snapshot_step = 500,show_metric = True, run_id="CatVSDog")