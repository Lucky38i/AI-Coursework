from multiprocessing import freeze_support

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator, load_img

if __name__ == '__main__':
    train_dir = "data/fruits-360_dataset/fruits-360/Training"
    test_dir = "data/fruits-360_dataset/fruits-360/Test"
    batch_size = 128
    epochs = 30

    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rescale=1 / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest',
                                       validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100))
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
    label_map = test_generator.class_indices
    class_num = train_generator.num_classes
    image_shape = test_generator.image_shape

    model = Sequential()
    """ CNN From Anindita Pani
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
    """


    model.add(Conv2D(filters=8, kernel_size=(5, 5), padding="Same", activation="relu", input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=16, kernel_size=(4, 4), padding="Same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=(4, 4), padding="Same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation="softmax"))

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    

    checkpointer = ModelCheckpoint(filepath='data/models/stack_cnn.hdf5', verbose=1, save_best_only=True)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=30,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_data=test_generator,
                        validation_steps=test_generator.samples // batch_size,
                        shuffle=True)

    score = model.evaluate_generator(test_generator, verbose=1)
    print('Test Accuracy: ', score[1])
