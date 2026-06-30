import os
import cv2
import numpy as np

IMG_SIZE = 128
MAX_SAMPLES = 500  # Limit samples for faster training

def load_data():
    data = []
    labels = []

    images_path = "data/images"
    masks_path = "data/masks"

    # Get all image files and sort them
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
    
    # Limit to MAX_SAMPLES
    image_files = image_files[:MAX_SAMPLES]
    print(f"Loading {len(image_files)} images...")
    
    for i, img_name in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing {i}/{len(image_files)}...")
            
        img_path = os.path.join(images_path, img_name)
        mask_path = os.path.join(masks_path, img_name)

        try:
            # Load image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
                mask = mask / 255.0  # Normalize to 0-1
            else:
                mask = np.zeros((IMG_SIZE, IMG_SIZE))  # Default to no tumor
            
            labels.append(mask)
            
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            pass

    X = np.array(data, dtype=np.float32) / 255.0
    y = np.array(labels)

    if y.ndim == 3:
        y = np.expand_dims(y, axis=-1)

    tumor_image_count = np.count_nonzero(np.sum(y, axis=(1, 2, 3)) > 0)
    no_tumor_image_count = len(y) - tumor_image_count

    print(f"Loaded {len(X)} images")
    print(f"Tumor images: {tumor_image_count}, No tumor images: {no_tumor_image_count}")
    
    return X, y