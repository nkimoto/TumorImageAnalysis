import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import glob
import matplotlib.pyplot as plt
import time
import os


STAIN = 'ER'

# Load image dict
with open(f"im_dict_{STAIN}.pickle", "rb") as rf:
    im_dict = pickle.load(rf)

# Create case, control list
case = []
for i in im_dict["case"].values():
    case.extend(i)
len(case)

control = []
for i in im_dict["control"].values():
    control.extend(i)
len(control)

# Create data_x, data_y
data_x = []
data_y = []
data_x.extend(case)
data_y.extend([1] * len(case))
data_x.extend(control)
data_y.extend([0] * len(control))
assert len(data_x) == len(data_y)


# np.arrayに変換
data_x = np.array(data_x)
data_y = np.array(data_y)

# 学習用データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# 正規化
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot encording
# y_train = to_categorical(y_train, num_classes=2)
# y_test = to_categorical(y_test, num_classes=2


# Check train image size
print("X_train: ")
print(x_train.shape)
print("X_test: ")
print(x_test.shape)
print("y_train: ")
print(y_train.shape)
print("y_test: ")
print(y_test.shape)


# モデルの構築
# # Initialising the CNN
# classifier = Sequential()
# 
# # Step 1 - Convolution
# classifier.add(Conv2D(32, (3, 3), input_shape = x_train.shape[1:], activation = 'relu'))
# 
# # Step 2 - Pooling
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# # Adding a second convolutional layer
# classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# # Adding a third convolutional layer
# classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# # Adding a fourth convolutional layer
# classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# 
# # Step 3 - Flattening
# classifier.add(Flatten())
# 
# # Step 4 - Full connection
# classifier.add(Dense(units = 64, activation = 'relu'))
# classifier.add(Dense(units = 1, activation = 'sigmoid'))
# 
# # Compiling the CNN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 畳み込みニューラルネットワーク
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', 
                     input_shape=x_train.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), 
                      metrics=['acc'])

plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)


filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Optimize model
results = {}
epochs = 20
filepath = f"best_model_{STAIN}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit(x_train, y_train,
                         epochs = epochs,
                         validation_split=0.2,
                         callbacks = [checkpoint])
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
name = f'val_acc_{STAIN}.jpg'
plt.savefig(name, bbox_inches='tight')

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
name = f'val_loss_{STAIN}.jpg'
plt.savefig(name, bbox_inches='tight')
