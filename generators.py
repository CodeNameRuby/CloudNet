# generators.py
import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

class DataGenerator(Sequence):
    def __init__(self, image_dir, df, batch_size, input_shape, shuffle=True):
        self.image_dir = image_dir
        self.df = df
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.df.iloc[batch_indexes]

        X = []
        y = []

        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.image_dir, row['image'])
            mask_path = os.path.join(self.image_dir, row['mask'])

            img = load_img(img_path, target_size=self.input_shape)
            mask = load_img(mask_path, target_size=self.input_shape, color_mode='grayscale')

            img = img_to_array(img) / 255.0
            mask = img_to_array(mask) / 255.0

            X.append(img)
            y.append(mask)

        return np.array(X), np.array(y)


def get_train_val_generators(train_csv, val_csv, train_img_dir, val_img_dir, batch_size, input_shape):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_gen = DataGenerator(train_img_dir, train_df, batch_size, input_shape[:2])
    val_gen = DataGenerator(val_img_dir, val_df, batch_size, input_shape[:2], shuffle=False)

    steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
    val_steps = int(np.ceil(len(val_df) / batch_size))

    return train_gen, val_gen, steps_per_epoch, val_steps


class TestGenerator(Sequence):
    def __init__(self, image_dir, df, batch_size, input_shape):
        self.image_dir = image_dir
        self.df = df
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.image_names = df['image'].tolist()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_data = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []

        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.image_dir, row['image'])
            img = load_img(img_path, target_size=self.input_shape)
            img = img_to_array(img) / 255.0
            X.append(img)

        return np.array(X), np.zeros((len(X), *self.input_shape[:2], 1))  # dummy y


def get_test_generator(test_csv, test_img_dir, batch_size, input_shape):
    df = pd.read_csv(test_csv)
    gen = TestGenerator(test_img_dir, df, batch_size, input_shape[:2])
    return gen, len(df), df['image'].tolist()
