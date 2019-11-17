from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np

test_dir = "data/fruits-360_dataset/fruits-360/Test"
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
label_map = test_generator.class_indices

model = load_model("data/models/stack_cnn.hdf5")

img = image.load_img("data/upload/pomegranate.jpg", target_size=(100, 100))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)


images = np.vstack([y])
classes = model.predict_classes(images)
for label, num in label_map.items():
    if num == classes:
        print("I think this is a:", label)
