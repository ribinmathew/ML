import numpy as np
import tqdm
test_data = np.load('test_data.npy')
with open('submission_file.csv', 'w') as f:
    f.write('id,label\n')

with open('submission_file.csv', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num, model_out[1]))