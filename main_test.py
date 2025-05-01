# main_test.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from generators import get_test_generator
from losses import dice_loss

# Parameters
BATCH_SIZE = 1
INPUT_SHAPE = (256, 256, 3)  # Same as in training
THRESHOLD = 0.5  # For binary mask
TEST_CSV = 'data/test.csv'
TEST_IMG_DIR = 'data/test_images'
SAVE_DIR = 'predictions'
MODEL_PATH = 'checkpoints/best_model.h5'

os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss})
print(f"Loaded model from {MODEL_PATH}")

# Load test data
test_gen, test_steps, test_image_names = get_test_generator(
    TEST_CSV, TEST_IMG_DIR, BATCH_SIZE, INPUT_SHAPE
)

# Predict and save
print("Generating predictions...")
for i in range(test_steps):
    x_batch, _ = next(test_gen)
    pred_mask = model.predict(x_batch)[0, :, :, 0]  # Assuming (1, H, W, 1)
    pred_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255
    image_name = test_image_names[i]

    save_path = os.path.join(SAVE_DIR, os.path.splitext(image_name)[0] + '_mask.png')
    Image.fromarray(pred_mask).save(save_path)

    print(f"Saved prediction to: {save_path}")

print("All predictions saved.")
