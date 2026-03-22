import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# 🔹 Chemin vers le dataset (grand dossier contenant les sous-dossiers de classes)
dataset_dir = r"C:\Users\21628\Downloads\esp32CAM\dataset" # mettre le chemin réel

# 🔹 Paramètres
img_height, img_width = 224, 224
batch_size = 32
epochs = 15  # tu peux augmenter selon le dataset

# 🔹 Data augmentation avec split pour validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# 🔹 Détecter automatiquement le nombre de classes
num_classes = len(train_generator.class_indices)
print(f"🔹 Classes détectées : {train_generator.class_indices}")

# 🔹 Construire le modèle CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # automatiquement selon le dataset
])

# 🔹 Compiler le modèle
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🔹 Entraîner le modèle
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# 🔹 Sauvegarder le modèle
model.save("plant_model.h5")
print("✅ Modèle sauvegardé dans plant_model.h5")

# 🔹 Affichage des courbes d'entraînement
plt.figure(figsize=(12,5))

# Précision
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Courbe de précision')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Courbe de loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()