from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# 🔹 Dossier pour stocker toutes les images
UPLOAD_FOLDER = r"C:\Users\21628\Downloads\esp32CAM\image_test"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 Route d'accueil
@app.route('/')
def index():
    return "Server is running! Use /upload to POST an image."

# 🔹 Route pour recevoir et stocker l'image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # 🔹 Sauvegarder l'image avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    return jsonify({
        "message": "Image received and stored successfully!",
        "filename": filename
    })

# 🔹 Lancer le serveur
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)