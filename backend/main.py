from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Inisialisasi Flask
app = Flask(_name_)

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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Membaca file gambar
        image = Image.open(file.file)
        
        # Preprocess gambar
        processed_image = preprocess_image(image)

        # Lakukan prediksi
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions)

        # Daftar nama kelas dataset
        class_names = ['Scabies', 'Dermatitis', 'Lepra', 'Health']
        result = class_names[predicted_class]

        return {"prediction": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
