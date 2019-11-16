from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator, load_img


train_dir = "data/fruits-360_dataset/fruits-360/Training"
test_dir = "data/fruits-360_dataset/fruits-360/Test"
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100))
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
label_map = test_generator.class_indices

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, input_shape=(100, 100, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(120, activation='softmax'))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

batch_size = 32
epochs = 30

checkpointer = ModelCheckpoint(filepath='cnn.hdf5', verbose=1, save_best_only=True)


model.fit_generator(train_generator,
                    epochs=30,
                    steps_per_epoch=train_generator.samples / batch_size,
                    validation_steps=test_generator.samples / batch_size,
                    validation_data=test_generator,
                    callbacks=[checkpointer],
                    verbose=1, shuffle=True)


score = model.evaluate_generator(test_generator, verbose=1)
print('Test Accuracy: ', score[1])

