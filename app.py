from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import jinja2

app = Flask(__name__)

dic = {0: 'Normal', 1: 'Pneumonia'}

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
jinja_enviroment = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))


def predict_label_vgg(img_path, model):
    imagen = Image.open(img_path).convert('RGB').resize((200, 200))
    imagen_normalizada = np.array(imagen) / 255.0
    imagen_normalizada = imagen_normalizada.reshape(1, 200, 200, 3)
    prediccion_vgg = model.predict(imagen_normalizada)

    return dic[round(prediccion_vgg[0][0])]


def predict_label_v3(img_path, model):
    imagen = Image.open(img_path).convert('RGB').resize((150, 150))
    imagen_normalizada = np.array(imagen) / 255.0
    imagen_normalizada = imagen_normalizada.reshape(1, 150, 150, 3)
    prediccion_v3 = model.predict(imagen_normalizada)

    return dic[round(prediccion_v3[0][0])]


def predict_label_resnet(img_path, model):
    imagen = Image.open(img_path).convert('RGB').resize((200, 200))
    imagen_normalizada = np.array(imagen) / 255.0
    imagen_normalizada = imagen_normalizada.reshape(1, 200, 200, 3)
    prediccion_resnet = model.predict(imagen_normalizada)

    return dic[round(prediccion_resnet[0][0])]

def predict_label_nopreentrenada1(img_path, model):
    imagen = Image.open(img_path).convert('L').resize((150, 150))
    imagen_normalizada = np.array(imagen) / 255.0
    imagen_normalizada = imagen_normalizada.reshape(1, 150,150, 1)
    prediccion_nopreentreanada1 = model.predict(imagen_normalizada)

    return dic[round(prediccion_nopreentreanada1[0][0])]

def predict_label_nopreentrenada2(img_path, model):
    imagen = Image.open(img_path).convert('L').resize((150, 150))
    imagen_normalizada = np.array(imagen) / 255.0
    imagen_normalizada = imagen_normalizada.reshape(1, 150,150, 1)
    prediccion_nopreentreanada2 = model.predict(imagen_normalizada)

    return dic[round(prediccion_nopreentreanada2[0][0])]


def modelo(img_path):
    model_vgg = tf.keras.models.load_model('vgg-16.h5')
    model_vgg.make_predict_function()
    prediccion_vgg = predict_label_vgg(img_path, model_vgg)

    model_inc = tf.keras.models.load_model('InceptionV3.h5')
    model_inc.make_predict_function()
    prediccion_inc = predict_label_v3(img_path, model_inc)

    model_resnet50 = tf.keras.models.load_model('resnet50.h5')
    model_resnet50.make_predict_function()
    prediccion_resnet50 = predict_label_resnet(img_path, model_resnet50)

    model_np1 = tf.keras.models.load_model('nopreentrenada.h5')
    model_np1.make_predict_function()
    prediccion_np1 = predict_label_nopreentrenada1(img_path, model_np1)

    model_np2 = tf.keras.models.load_model('nopreentrenada2.h5')
    model_np2.make_predict_function()
    prediccion_np2 = predict_label_nopreentrenada2(img_path, model_np2)

    return (prediccion_vgg, prediccion_inc, prediccion_resnet50,prediccion_np1,prediccion_np2)


@app.route("/", methods=['GET', 'POST'])
def main():
    template = jinja_enviroment.get_template('index.html')
    return template.render()


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        my_id = request.form.get("boton", "")
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)

        (p_vgg,p_inc,p_rn,p_np1,p_np2) = modelo(img_path)


    template = jinja_enviroment.get_template('imagen.html')
    # return template.render(prediction=p, img_path=img_path, red=my_id)
    return template.render(prediction=p_vgg, prediction_inc=p_inc, prediction_rn=p_rn,prediction_np1=p_np1,
                           prediction_np2=p_np2,
                           img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
