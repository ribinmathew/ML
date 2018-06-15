import pandas as pd

data = pd.read_csv("Whale_image_label")
#print(data["Label"])

"so we have loaded the data which contain numpy image and the label"
"now we have to do hot encoding"
"thats a big challenge"
"there are 4251 unique ids, this is our target variable, so we will need a one hot encoding,which can be achived by get dummies() method in pandas"

train_data = pd.concat([data,pd.get_dummies(data.Label)], axis=1)
print(train_data)
#print(train_data)
"so  we have done one hot encoding using get_dummies() method in pandas"
"now we are gonna save the train_Data file which is done with onehot encoding"
train_data.to_csv("train_data",index=False)