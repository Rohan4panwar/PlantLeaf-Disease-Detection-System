from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import os
import tensorflow as tf

# Create Flask app
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    r"C:\Users\Rohan Panwar\OneDrive\Desktop\plant\model\plant_cnn.keras"
)

# Class labels
label = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
         'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
         'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
         'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
         'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
         'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
         'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
         'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
         'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
         'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
         'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load disease descriptions
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

# Route to serve uploaded images
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

# Home page
@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# Preprocessing function
def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

# Prediction function
def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

# Upload image and predict
@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        
        # Ensure upload folder exists
        upload_folder = "uploadimages"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filename = f"temp_{uuid.uuid4().hex}_{image.filename}"
        save_path = os.path.join(upload_folder, filename)
        image.save(save_path)

        prediction = model_predict(save_path)
        return render_template('home.html', result=True,
                               imagepath=f'/uploadimages/{filename}',
                               prediction=prediction)
    else:
        return redirect('/')

# Start Flask app in notebook cell
app.run(port=5000, debug=False, use_reloader=False)
