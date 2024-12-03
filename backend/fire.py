# Import library
from flask import Flask, request, jsonify   # Framework untuk API
import tensorflow as tf                     # Library Machine Learning
import numpy as np                          # Library untuk bagian perhitungan
from pathlib import Path                    # Library membaca file
from PIL import Image                       # Library untuk mengatur gambar
import json                                 # Library untuk load file json
import io                                   # Library untuk bagian Input/Ouput

app = Flask(__name__)

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label class dari dataset
label_mapping = {0: "Flea_allergy", 1: "Health", 2: "Ringworm", 3: "Scabies"}

# Preprocessing gambar agar sesuai ke TFLite model
def preprocess_image(image_file):
    image = Image.open(image_file).resize((224, 224)) 
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)  
    return input_data

# Load dataset artikel json
def from_articles():
    article_path = Path("kumpulan-artikel/artikel.json")
    with open(article_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    try:
        with open(article_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: Artikel tidak ditemukan di {article_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON format di {article_path}")
        return {}
    
# Load article saat prediksi dimulai
cat_disease_articles = from_articles()
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files: # Cek apakah ada gambar yang diupload
        return jsonify({"error": "Tidak ada gambar yang diupload"}), 400

    # Mengupload file gambar
    file = request.files['image']
    try:
        input_data = preprocess_image(file)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        predicted_label = label_mapping.get(predicted_class, "Unknown")
        articles = cat_disease_articles.get(predicted_label, {
            "title": "Tidak Diketahui",
            "content": "Maaf, tidak ada informasi untuk kondisi ini."
        })

        # Prediksi yang dihasilkan beserta probabilitas-nya dan juga artikel yang sesuai
        return jsonify({
            "predicted_label": predicted_label,
            "raw_output": output_data.tolist(),
            "article": articles
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk mengecek status server
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server berjalan dengan baik'})

if __name__ == '__main__':
    app.run(debug=True)
