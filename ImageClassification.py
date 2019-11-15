import numpy
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint

seed = 21
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3,3), input_shape=X_train.shape[1:],
                 padding='same', activation='relu'))

# Drop 20% of existing connection to preventing overfitting
model.add(Dropout(0.2))

# Normalise inputs going into next layer
model.add(BatchNormalization())

# Second Convolutional Layer (Increased Filter) to learn more complex representations
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))

# Add Pooling layer to abstract unnecessary parts of image with max pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Third Convlutional Layer (Increased Filter) to learn more complex representations
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# Flatten the data to a vector to be processed
model.add(Flatten())
model.add(Dropout(0.2))

# 1st Dense layer with kernel constraint to prevent overfitting
model.add(Dense(256, kernel_constraint=maxnorm(3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# 2nd Dense layer with kernel constraint to prevent overfitting
model.add(Dense(128, kernel_constraint=maxnorm(3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# 3rd Dense layer representing number of classes with softmax to select high probability as output
model.add(Dense(class_num, activation='softmax'))

epochs = 10
optimizer = 'adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


numpy.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks_list)
score = model.evaluate(X_test, y_test, verbose=0)
print("ACCURACY: %.2f%%" % (score[1]*100))

''' Uncomment this to continue model
load_model = load_model("model.h5")
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
load_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks_list)
'''