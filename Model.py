import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("📂 Loading preprocessed data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

num_classes = y_train.shape[1]  
print(f"✅ Data loaded successfully! Classes: {num_classes}")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("🔁 Setting up data augmentation...")

datagen = ImageDataGenerator(
    rotation_range=10,        
    zoom_range=0.1,           
    width_shift_range=0.1,    
    height_shift_range=0.1,   
    shear_range=0.1,          
    fill_mode='nearest'     
)

datagen.fit(X_train)

train_generator = datagen.flow(
    X_train, y_train,
    batch_size=32,
    shuffle=True
)

print("✅ Data augmentation ready!\n")
print("🧠 Building CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.25),

   
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("\n🚀 Starting training...")
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\n✅ Training complete! Best model saved as 'best_model.keras'")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📈 Test Accuracy: {test_acc * 100:.2f}%")
print(f"❌ Test Loss: {test_loss:.4f}")
