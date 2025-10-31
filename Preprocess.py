import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_DIR = 'D:/Coding/MATH_DIGIT_RECOGNIZER/Dataset'  
IMAGE_SIZE = (64, 64)  
NUM_CHANNELS = 1       
TEST_SPLIT = 0.2

images = []
labels = []

class_names = os.listdir(DATA_DIR)
class_names = [c for c in class_names if os.path.isdir(os.path.join(DATA_DIR, c))]
print(f"Found {len(class_names)} classes: {class_names}")

for label_index, class_name in enumerate(class_names):
    class_path = os.path.join(DATA_DIR, class_name)
    image_files = os.listdir(class_path)

    for image_name in image_files:
        img_path = os.path.join(class_path, image_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            img = cv2.resize(img, IMAGE_SIZE)

            images.append(img)
            labels.append(label_index)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

images = np.array(images, dtype="float32") / 255.0  
labels = np.array(labels, dtype="int")

images = np.expand_dims(images, axis=-1)

labels_categorical = to_categorical(labels, num_classes=len(class_names))

print(f"Total images processed: {len(images)}")
print(f"Image shape: {images.shape}")
print(f"Label shape: {labels_categorical.shape}")


X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels_categorical, test_size=TEST_SPLIT, random_state=42, stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("✅ Dataset creation completed and saved as .npy files.")