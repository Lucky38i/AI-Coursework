from keras import Sequential
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from PIL import Image

test_dir = "data/fruits-360_dataset/fruits-360/Test"
checkpoint_file = 'data/models/pani_rmsprop_cnn.hdf5'

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
label_map = test_generator.class_indices


model = load_model(checkpoint_file)
model.load_weights(checkpoint_file)

img = image.load_img("data/upload/pineapple1.jpg", target_size=(100, 100))
img_tensor = np.array(img).astype('float32')/255
img_tensor = np.expand_dims(img_tensor, axis=0)

classes = model.predict_classes(img_tensor)
for label, num in label_map.items():
    if num == classes:
        print("I think this is a:", label)
