from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# load model TFLite
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image = image.convert('RGB')
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Definisikan kelas penyakit kucing
cat_disease_classes = ["Health", "Ringworm", "Scabies", "Sporotrichosis"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cek apakah ada file gambar yang dikirim
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada gambar yang dikirim'}), 400
        
        # Ambil file gambar
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocessing gambar
        processed_image = preprocess_image(image)
        
        # Load model dan dapatkan input/output tensors
        interpreter = load_tflite_model()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Jalankan inferensi
        interpreter.invoke()
        
        # Dapatkan hasil prediksi
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = cat_disease_classes[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Siapkan response
        response = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(cat_disease_classes, predictions[0])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint untuk mengecek status server
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server berjalan dengan baik'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)