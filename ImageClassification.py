import os.path

from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

# TODO Determine a viable model to produce useful results on unseen data
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


if __name__ == '__main__':
    train_dir = "data/fruits-360_dataset/fruits-360/Training"
    test_dir = "data/fruits-360_dataset/fruits-360/Test"
    checkpoint_file = 'data/models/pani_adam_cnn.hdf5'
    batch_size = 32
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

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100))
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
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
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        model.summary()

    checkpointer = ModelCheckpoint(filepath=checkpoint_file, verbose=1, save_best_only=True, save_weights_only=False)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_data=test_generator,
                        validation_steps=test_generator.samples // batch_size,
                        shuffle=True)

    score = model.evaluate_generator(test_generator, verbose=1)
    print('Test Accuracy: ', score[1])
