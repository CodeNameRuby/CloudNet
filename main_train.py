# main_train.py
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from models import cloud_net
from generators import get_train_val_generators
from losses import dice_loss

# GPU growth config (Optional but helps avoid CUDNN errors)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parameters
BATCH_SIZE = 4
EPOCHS = 50
INPUT_SHAPE = (256, 256, 3)  # Resize your 930x930 RGB images to this

# Paths
train_csv = 'data/train.csv'
val_csv = 'data/val.csv'
train_img_dir = 'data/train_images'
val_img_dir = 'data/val_images'
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Load data generators
train_gen, val_gen, steps_per_epoch, val_steps = get_train_val_generators(
    train_csv, val_csv, train_img_dir, val_img_dir, BATCH_SIZE, INPUT_SHAPE
)

# Model
model = cloud_net(INPUT_SHAPE)
model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=['accuracy'])

# Checkpoint
checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=[checkpoint]
)
