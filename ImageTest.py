import numpy as np
from keras.models import load_model
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, img_to_array


def preprocess_image(file_dir):
    temp_img = image.load_img(file_dir, target_size=(100, 100))
    temp_img = img_to_array(temp_img)
    temp_img = temp_img.astype('float') / 255
    temp_img = np.expand_dims(temp_img, axis=0)

    return temp_img


if __name__ == '__main__':
    test_dir = "data/fruits-360_dataset/fruits-360/Test"
    checkpoint_file = 'data/models/pani_rmsprop_cnn.hdf5'

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
    label_map = test_generator.class_indices

    model = load_model(checkpoint_file)

    img_tensor = preprocess_image("data/upload/bananas.jpg")

    classes = model.predict_classes(img_tensor)
    for label, num in label_map.items():
        if num == classes:
            print("I think this is a:", label)
