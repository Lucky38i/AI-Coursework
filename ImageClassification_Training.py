import os.path

import kaggle
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import load_model
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, img_to_array


def build_model():
    temp_model = Sequential()

    temp_model.add(Conv2D(filters=16, kernel_size=3, input_shape=image_shape, padding='same'))
    temp_model.add(Activation('relu'))
    temp_model.add(MaxPooling2D(pool_size=2))

    temp_model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    temp_model.add(MaxPooling2D(pool_size=2))

    temp_model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    temp_model.add(MaxPooling2D(pool_size=2))

    temp_model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
    temp_model.add(MaxPooling2D(pool_size=2))
    temp_model.add(Dropout(0.3))

    temp_model.add(Flatten())

    temp_model.add(Dense(128, activation='relu'))
    temp_model.add(Dropout(0.4))

    temp_model.add(Dense(class_num, activation='softmax'))

    return temp_model


def download_data():
    kaggle.api.authenticate()

    dataset = 'moltean/fruits'
    downloaddir = 'data/'

    kaggle.api.dataset_download_files(dataset, path=downloaddir, unzip=True)


def preprocess_image(file_dir):
    temp_img = image.load_img(file_dir, target_size=(100, 100))
    temp_img = img_to_array(temp_img)
    temp_img = temp_img.reshape((1,) + temp_img.shape)
    temp_img = temp_img / 255.

    return temp_img


if __name__ == '__main__':
    # Uncomment if you wish to download the dataset
    # download_data()
    train_dir = "data/fruits-360_dataset/fruits-360/Training"
    test_dir = "data/fruits-360_dataset/fruits-360/Test"
    checkpoint_file = 'data/models/pani_adam_200_cnn.hdf5'
    batch_size = 32
    epochs = 200

    # Image data generator with varying parameters to ensure skewed pictures
    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rescale=1 / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest',
                                       validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100),
                                                        batch_size=batch_size, shuffle=True)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100),
                                                      batch_size=batch_size, shuffle=True)
    label_map = test_generator.class_indices
    class_num = train_generator.num_classes
    image_shape = test_generator.image_shape

    # Load existing model if it exists otherwise build new model
    if os.path.isfile(checkpoint_file):
        print("Loading existing model")
        model = load_model(checkpoint_file)
        model.summary()
    else:
        print("Creating new model")
        model = build_model()
        # Compile using categorical as we are classifying images, adam optimizer as its most popular
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        model.summary()

    # Callbacks to save the best model only
    callbacks = [ModelCheckpoint(filepath=checkpoint_file, verbose=1, save_best_only=True, save_weights_only=False)]

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=test_generator,
                        validation_steps=test_generator.samples // batch_size,
                        shuffle=True)

    score = model.evaluate_generator(test_generator, verbose=1)
    print('Test Accuracy: ', score[1])
