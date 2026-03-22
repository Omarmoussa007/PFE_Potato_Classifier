import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from datetime import datetime
import base64

# 🔹 Charger le modèle
model_path = r"C:\Users\21628\Downloads\esp32CAM\plant_model.h5"
model = load_model(model_path)

# 🔹 Classes
class_labels = {0: 'Potato___Early_blight',
                1: 'Potato___Late_blight',
                2: 'Potato___Healthy'}

# 🔹 Conseils avec cause et couleur
advice = {
    "Potato___Early_blight": {
        "description": "La pomme de terre présente des symptômes de mildiou précoce.",
        "cause": "Champignon Alternaria solani, souvent dû à l'humidité et chaleur modérée.",
        "help": "Retirez les feuilles infectées, appliquez un fongicide adapté et évitez l'humidité excessive.",
        "color": "#FF4136",
        "emoji": "🔴"
    },
    "Potato___Late_blight": {
        "description": "La pomme de terre présente des symptômes de mildiou tardif.",
        "cause": "Phytophthora infestans, favorisé par humidité élevée et pluie.",
        "help": "Éliminez immédiatement les plantes infectées, traitez rapidement avec un fongicide et surveillez les autres plants.",
        "color": "#FF851B",
        "emoji": "🟠"
    },
    "Potato___Healthy": {
        "description": "La pomme de terre est saine.",
        "cause": "Aucune maladie détectée.",
        "help": "Continuez les soins normaux : arrosage régulier, fertilisation et surveillance des maladies.",
        "color": "#2ECC40",
        "emoji": "🟢"
    }
}

# 🔹 Taille des images
img_height, img_width = 224, 224

# 🔹 Dossier uploads
UPLOAD_FOLDER = r"C:\Users\21628\Downloads\esp32CAM\image_test"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# 🔹 Page config et titre
st.set_page_config(
    page_title="Détection des maladies de la pomme de terre",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center; color:#2E86C1;'>🥔 Détection des maladies de la pomme de terre</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#7F8C8D;'>Analyse automatique des images et conseils pour vos cultures</p>",
    unsafe_allow_html=True
)

# 🔹 Sidebar
st.sidebar.header("Options")
filter_class = st.sidebar.selectbox("Filtrer par classe", ["Toutes"] + list(class_labels.values()))
uploaded_files = st.sidebar.file_uploader(
    "📤 Ajouter une ou plusieurs images",
    type=['jpg','jpeg','png'],
    accept_multiple_files=True
)

# 🔹 Lister toutes les images existantes
all_images = []
for root, dirs, files in os.walk(UPLOAD_FOLDER):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_images.append(os.path.join(root, file))
st.sidebar.write(f"Nombre d'images trouvées: {len(all_images)}")

# 🔹 Bouton pour supprimer toutes les images
if st.sidebar.button("🗑️ Supprimer toutes les images"):
    for img_path in all_images:
        try:
            os.remove(img_path)
        except Exception as e:
            st.warning(f"Impossible de supprimer {img_path}: {e}")
    st.success("✅ Toutes les images ont été supprimées. Rechargez la page pour actualiser la liste.")

# 🔹 Fonction de prédiction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    class_name = class_labels[class_index]
    confidence = float(np.max(pred) * 100)
    return class_name, confidence

# 🔹 Affichage des images stockées en cards
st.subheader("📸 Images stockées")
cols = st.columns(3)
for i, img_path in enumerate(all_images):
    class_name, confidence = predict_image(img_path)
    if filter_class != "Toutes" and class_name != filter_class:
        continue
    with cols[i % 3]:
        st.image(img_path, use_column_width=True)
        color = advice[class_name]['color']
        emoji = advice[class_name]['emoji']
        st.markdown(
            f"<div style='border:2px solid {color}; padding:10px; border-radius:10px;'>"
            f"<span style='color:{color}; font-weight:bold'>{emoji} {class_name} ({confidence:.1f}%)</span><br>"
            f"**Description:** {advice[class_name]['description']}<br>"
            f"**Cause possible:** {advice[class_name]['cause']}<br>"
            f"**Conseils:** {advice[class_name]['help']}"
            f"</div>", unsafe_allow_html=True)
        st.markdown("---")

# 🔹 Traitement des images uploadées
if uploaded_files:
    st.subheader("🆕 Nouvelles images uploadées")
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    for i, uploaded_file in enumerate(uploaded_files):
        # sauvegarder l'image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{uploaded_file.name}")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # prédiction
        class_name, confidence = predict_image(save_path)
        color = advice[class_name]['color']
        emoji = advice[class_name]['emoji']
        
        st.image(uploaded_file, use_column_width=True)
        st.markdown(
            f"<div style='border:2px solid {color}; padding:10px; border-radius:10px;'>"
            f"<span style='color:{color}; font-weight:bold'>{emoji} {class_name} ({confidence:.1f}%)</span><br>"
            f"**Description:** {advice[class_name]['description']}<br>"
            f"**Cause possible:** {advice[class_name]['cause']}<br>"
            f"**Conseils:** {advice[class_name]['help']}"
            f"</div>", unsafe_allow_html=True)
        
        progress_bar.progress((i+1)/total_files)
        st.balloons()  # 🎈 animation