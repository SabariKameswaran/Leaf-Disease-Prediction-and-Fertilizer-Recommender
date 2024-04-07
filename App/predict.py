import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from flask import render_template
import io


class plantleaf:
    def predictPlantImage(self, image_file):

        self.image_file = image_file
        model = load_model("C:\\Users\\SABARI HARI\\Downloads\\PlantDiseasePrediction\\App\\COFFEE_model.h5")
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(self.image_file.filename)) 
        self.image_file.save(file_path)
        test_image = image.load_img(file_path, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0) 
        preds = model.predict(test_image)
        preds = np.argmax(preds,axis=1)  # The numpy. argmax() function returns indices of the max element of the array in a particular axis.
        #print(preds)

        if preds == 0:
            prediction = "The Disease is Cerscospora. The Fertilizers are Penthiopyrad,Strobilurin,Copper Fungicides"
            return prediction
        elif preds == 1:
            prediction = "Healthy Leaf"
            return prediction
        elif preds == 2:
            prediction = "The Disease is Leaf Rust. The Fertilizers are RANSOM,GREENCOP,DEFACTO"
            return prediction
        elif preds == 3:
            prediction = "The Disease is Miner. The Fertilizers are Sevin, malathion, and lindane."
            return prediction
        else:
            prediction = "The Disease is Phoma. The Fertilizers propiconazole and difenoconazole, and premixtures, pydiflumetofen + fludioxonil or pydiflumetofen + difenoconazol"
            return prediction