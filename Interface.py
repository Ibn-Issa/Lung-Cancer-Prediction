#import dependencies
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os

#load trained model
model = tf.keras.models.load_model('./trained_lung_cancer_model.keras')


#initialize flask
app = Flask(__name__)

#get path to uploads folder
upload_folder = os.path.join('static', 'uploads')

app.config['UPLOAD'] = upload_folder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/system')
def system():
    return render_template('system.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST', 'GET'])
def diagonize():
    if request.method == 'POST':
        #get image
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)

        test_image = tf.keras.preprocessing.image.load_img(img, target_size=(350, 350))
        # test_image = tf.keras.preprocessing.image.img_to_array(test_image) / 255
        test_image = np.expand_dims(test_image, axis = 0)
        
        #make prediction
        y_pred = model.predict(test_image)
        result = np.argmax(y_pred)

        if result == 0:
            prediction = [img, "This is a case of Lung ACA"]
            return render_template('system.html', prediction=prediction)
        elif result == 1:
            prediction = [img, "This is a Normal Lung"]
            return render_template('system.html', prediction=prediction)
        else:
            prediction = [img, "This is a case of Lung SCC"]
            return render_template('system.html', prediction=prediction)

    

if __name__ == '__main__':
    app.run(debug=True)