from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model

if __name__ == '__main__':
    train_dir = "data/fruits-360_dataset/fruits-360/Training"
    test_dir = "data/fruits-360_dataset/fruits-360/Test"

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=False)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(100, 100), shuffle=True)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
    label_map = test_generator.class_indices
    class_num = train_generator.num_classes
    image_shape = test_generator.image_shape

    checkpointer = ModelCheckpoint(filepath='data/models/stack_cnn.hdf5', verbose=1, save_best_only=True)

    model = load_model("data/models/stack_cnn.hdf5")

    batch_size = 32
    epochs = 30

    model.fit_generator(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[checkpointer],
                        validation_data=test_generator,
                        validation_steps=test_generator.samples // batch_size,
                        shuffle=True)
