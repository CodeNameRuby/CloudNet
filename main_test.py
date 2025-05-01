# main_test.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from losses import dice_loss
from glob import glob

# Parameters
BATCH_SIZE = 1
INPUT_SHAPE = (256, 256, 3)
THRESHOLD = 0.5
TEST_IMG_DIR = 'Test/images'         # folder with test .tif images
MODEL_PATH = 'checkpoints/best_model.h5'
SAVE_DIR = 'predictions'

os.makedirs(SAVE_DIR, exist_ok=True)

# Load model
model = load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss})
print(f"✅ Model loaded from {MODEL_PATH}")

# Get test image paths
image_paths = sorted(glob(os.path.join(TEST_IMG_DIR, '*.tif')))
print(f"📸 Found {len(image_paths)} test images")

# Loop through and predict
for img_path in image_paths:
    img_name = os.path.basename(img_path)
    
    # Load and preprocess image
    img = load_img(img_path, target_size=INPUT_SHAPE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)

    # Predict
    pred_mask = model.predict(img_array)[0, :, :, 0]  # shape: (H, W)
    pred_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255

    # Save result
    save_path = os.path.join(SAVE_DIR, img_name.replace('.tif', '_mask.png'))
    Image.fromarray(pred_mask).save(save_path)
    print(f"💾 Saved: {save_path}")

print("✅ All predictions saved.")
