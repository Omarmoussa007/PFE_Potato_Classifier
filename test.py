import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 🔹 Chemin du modèle sauvegardé
model_path = r"C:\Users\21628\Downloads\esp32CAM\plant_model.h5"

# 🔹 Chemin de l'image à tester
img_path = r"C:\Users\21628\Downloads\esp32CAM\image_test\f2a56e13-438e-4bef-a3ea-beb9f347d481___RS_Early.B 6691.JPG"

# 🔹 Charger le modèle
model = load_model(model_path)
print("✅ Modèle chargé avec succès")

# 🔹 Taille des images
img_height, img_width = 224, 224

# 🔹 Labels des classes
class_labels = {0: 'Potato___Early_blight',
                1: 'Potato___Late_blight',
                2: 'Potato___Healthy'}

# 🔹 Charger et prétraiter l'image
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# 🔹 Prédiction
pred = model.predict(img_array)
class_index = np.argmax(pred)
class_name = class_labels[class_index]
confidence = np.max(pred) * 100

print(f"Image: {img_path} -> Prédiction: {class_name} ({confidence:.2f}%)")