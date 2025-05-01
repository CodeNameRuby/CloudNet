# generators.py
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from glob import glob
from sklearn.model_selection import train_test_split

class DataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, file_list, batch_size, input_shape, shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_list)

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        X, y = [], []

        for file_name in batch_files:
            img_path = os.path.join(self.image_dir, file_name)
            mask_path = os.path.join(self.mask_dir, file_name)

            # Load image and mask
            img = load_img(img_path, target_size=self.input_shape)
            mask = load_img(mask_path, target_size=self.input_shape, color_mode='grayscale')

            # Normalize and convert to arrays
            img = img_to_array(img) / 255.0
            mask = img_to_array(mask) / 255.0

            X.append(img)
            y.append(mask)

        return np.array(X), np.array(y)

def get_train_val_generators(image_dir, mask_dir, batch_size, input_shape, val_split=0.2):
    all_filenames = [os.path.basename(f) for f in glob(os.path.join(image_dir, '*.tif'))]
    train_files, val_files = train_test_split(all_filenames, test_size=val_split, random_state=42)

    train_gen = DataGenerator(image_dir, mask_dir, train_files, batch_size, input_shape[:2], shuffle=True)
    val_gen = DataGenerator(image_dir, mask_dir, val_files, batch_size, input_shape[:2], shuffle=False)

    steps_per_epoch = int(np.ceil(len(train_files) / batch_size))
    val_steps = int(np.ceil(len(val_files) / batch_size))

    return train_gen, val_gen, steps_per_epoch, val_steps
