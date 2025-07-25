from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('pet_classification.h5')  # model.h5 senin eğittiğin model dosyası
target_size = (224, 224)  # modeline göre ayarla

# Class 37
class_names =  ['Abyssinian' ,'american_bulldog' ,'american_pit_bull_terrier',
 'basset_hound' ,'beagle' ,'Bengal', 'Birman' ,'Bombay' ,'boxer',
 'British_Shorthair', 'chihuahua' ,'Egyptian_Mau', 'english_cocker_spaniel',
 'english_setter','german_shorthaired', 'great_pyrenees' ,'havanese',
 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon' ,'miniature_pinscher',
 'newfoundland', 'Persian', 'pomeranian', 'pug' ,'Ragdoll', 'Russian_Blue',
 'saint_bernard' ,'samoyed' ,'scottish_terrier' ,'shiba_inu' ,'Siamese',
 'Sphynx' ,'staffordshire_bull_terrier', 'wheaten_terrier',
 'yorkshire_terrier']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Dosya yükleme kontrolü
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Görseli modele uygun şekilde işleme
        img = image.load_img(filepath, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # normalize

        # Tahmin
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        return render_template('index.html', filename=filename, prediction=predicted_class)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    app.run(debug=True)
