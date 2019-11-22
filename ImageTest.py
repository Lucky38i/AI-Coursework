from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator

from ImageClassification import preprocess_image

if __name__ == '__main__':
    test_dir = "data/fruits-360_dataset/fruits-360/Test"
    checkpoint_file = 'data/models/pani_adam_200_cnn.hdf5'

    test_datagen = ImageDataGenerator(rescale=1 / 255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
    label_map = test_generator.class_indices

    model = load_model(checkpoint_file)

    img_tensor = preprocess_image("data/upload/apples1.jpg")

    classes = model.predict_classes(img_tensor, batch_size=1)
    for label, num in label_map.items():
        if num == classes:
            print("I think this is a:", label)

    score = model.evaluate_generator(test_generator, verbose=1)
    print('Test Accuracy: ', score[1])
