# main_train.py
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from models import cloud_net
from generators import get_train_val_generators
from losses import dice_loss

# Enable memory growth for GPU
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
INPUT_SHAPE = (256, 256, 3)  # Resize as needed

# Directories
image_dir = 'Train/images'
mask_dir = 'Train/masks'
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Load data generators
train_gen, val_gen, steps_per_epoch, val_steps = get_train_val_generators(
    image_dir, mask_dir, BATCH_SIZE, INPUT_SHAPE
)

# Build model
model = cloud_net(INPUT_SHAPE)
model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Train the model
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=[checkpoint]
)
