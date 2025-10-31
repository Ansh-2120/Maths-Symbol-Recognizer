# Math Symbol Recognizer 🔢

A deep learning project that recognizes handwritten mathematical symbols using a CNN model trained on the Kaggle dataset with 84 different mathematical symbols and characters.

## 📌 What It Does

This system recognizes handwritten mathematical symbols including:
- **Numbers**: 0-9
- **Greek Letters**: α, β, γ, δ, θ, λ, μ, σ, φ, etc.
- **Operators**: +, -, ×, =, ≠, ≤, ≥, <, >, ±
- **Functions**: sin, cos, tan, log, limit, sum, ∑
- **Special Symbols**: ∞, ∀, ∃, →, brackets, parentheses

## 🛠️ Tech Stack

### **Frameworks & Libraries**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV (cv2)** - Image preprocessing
- **NumPy** - Numerical operations
- **Scikit-learn** - Data splitting and metrics
- **Matplotlib** - Training visualization
- **KaggleHub** - Dataset download

### **Model Architecture**
- **Type**: Convolutional Neural Network (CNN)
- **Layers**: 
  - 3 Conv2D layers (32, 64, 128 filters)
  - BatchNormalization after each Conv layer
  - MaxPooling2D for feature reduction
  - Dropout (0.25, 0.5) for regularization
  - Dense layers (512 units) for classification
- **Input**: 64×64 grayscale images
- **Output**: 84 classes (softmax)

## ⚙️ How It Works

### **1. Data Preprocessing** (`Preprocess.py`)
```python
# Process flow:
Load images from 84 class folders
    ↓
Convert to grayscale
    ↓
Resize to 64×64 pixels
    ↓
Normalize pixel values (0-1)
    ↓
Train/Val/Test split (80/10/10)
    ↓
One-hot encode labels
    ↓
Save as .npy files
```

**Key Operations:**
- Image loading with `cv2.imread()`
- Grayscale conversion and resizing
- Normalization: `images / 255.0`
- Stratified splitting to maintain class distribution

### **2. Data Augmentation**
Real-time augmentation during training:
- **Rotation**: ±10°
- **Zoom**: ±10%
- **Width/Height Shift**: ±10%
- **Shear**: 0.1 intensity
- **Fill Mode**: Nearest neighbor

### **3. Model Training** (`Model.py`)

**Architecture Flow:**
```
Input (64×64×1)
    ↓
Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2)
    ↓
Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2)
    ↓
Conv2D(128, 3×3) → BatchNorm → MaxPool(2×2)
    ↓
Dropout(0.25)
    ↓
Flatten
    ↓
Dense(512, ReLU) → Dropout(0.5)
    ↓
Dense(84, Softmax)
```

**Training Configuration:**
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Epochs**: 25 (with early stopping)
- **Callbacks**: 
  - EarlyStopping (patience=5, monitor='val_loss')
  - ModelCheckpoint (save best model by 'val_accuracy')

### **4. Tech Flow Summary**

```
Dataset (Kaggle) 
    ↓
Preprocessing (OpenCV + NumPy)
    ↓
Data Augmentation (Keras ImageDataGenerator)
    ↓
CNN Training (TensorFlow/Keras)
    ↓
Model Evaluation (Test Set)
    ↓
Best Model Saved (best_model.keras)
```

## 📊 Dataset

- **Source**: Kaggle - Mathematical Symbols Dataset
- **Total Images**: ~7,000
- **Classes**: 84 different symbols
- **Format**: Grayscale PNG/JPG images
- **Image Size**: Variable (resized to 64×64)
- **Split**: 
  - Training: 5,579 images
  - Validation: 697 images
  - Test: 698 images

## 🚀 Training Process

The model was trained using **Google Colab with GPU acceleration**:

1. **Download Dataset** from Kaggle using KaggleHub
2. **Preprocess Images** - Grayscale, resize, normalize
3. **Split Data** - Train/Val/Test with stratification
4. **Setup Augmentation** - ImageDataGenerator for training
5. **Build CNN Model** - 3 Conv blocks + Dense layers
6. **Train with Callbacks** - Early stopping + checkpointing
7. **Evaluate** - Test set accuracy
8. **Save Model** - Best weights as `best_model.keras`

## 📁 Project Structure

```
├── Preprocess.py              # Data loading and preprocessing
├── Model.py                   # CNN model definition and training
├── math_recognizer.ipynb      # Complete pipeline (Google Colab)
├── best_model.keras           # Trained model weights
├── X_train.npy                # Training images
├── y_train.npy                # Training labels
├── X_val.npy                  # Validation images
├── y_val.npy                  # Validation labels
├── X_test.npy                 # Test images
└── y_test.npy                 # Test labels
```

## 🎯 Results

- **Test Accuracy**: ~67-83% (varies by run)
- **Classes Recognized**: 84 symbols
- **Model Size**: ~9.52 MB (2.5M parameters)
- **Training Time**: ~4-5 minutes on Colab GPU

## 💻 Requirements

```
tensorflow>=2.x
opencv-python
numpy
scikit-learn
matplotlib
kagglehub
```

## 🔮 Future Improvements

- Increase dataset size for better accuracy
- Add more complex augmentation techniques
- Implement ensemble models
- Create a web/mobile demo interface
- Real-time drawing recognition

---

**Built with deep learning to make math symbol recognition accessible! 🧮✨**

