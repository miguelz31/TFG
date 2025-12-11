# #
import numpy as np
import pandas as pd
import random

# folder
import os
import glob

# image x
from PIL import Image

# visu
import matplotlib.pyplot as plt
plt.rc('image', cmap='gray')

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import wrappers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

#AÑADIDO para borrar las carpetas que no me sirvan
import shutil

#Añadido para Matriz de Confusion

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import itertools
from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf
import gc

from tensorflow.keras.preprocessing.image import ImageDataGenerator


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.vgg16 import VGG16
# from keras.applications import vgg16
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import Accuracy


def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    # ax1.set_ylim(-0.1, 0.1)

    titulo1 = title + " loss"
    ax1.set_title(titulo1)

    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy' + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy' + exp_name)
    # ax2.set_ylim(0.9, 1.1)
    titulo2 = title + " Accuracy"
    ax2.set_title(titulo2)

    ax2.legend()
    return (ax1, ax2)


# plot_history(history, title='Model', axs=None, exp_name="");
# plot_history(history_model_pr, title ='Model_pr', axs = None, exp_name="");

# plot_history(history_mse, title ='Model_mse', axs = None, exp_name="");
# plot_history(history_model_pr_mse, title ='Model_pr_mse', axs = None, exp_name="");


def predecir(prediction, modelo, X_test, y_test):
    # AÑADIDO
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(8, 4)

    for row in range(0, 8):
        for col in range(0, 3):
            num_image = random.randint(0, X_test.shape[0])
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off');
            ax.set_title(
                "Predicted: " + categories[int(np.round(prediction)[num_image][0])] + " /\n True value: " + categories[
                    y_test[num_image]])
            # x = X_test[num_image]
            # result = x[:, :, 0]
            # # ax.imshow(result);
            ax.imshow(X_test[num_image]);
    fig.suptitle("Predicted label VS True label \n for the displayed chest X Ray pictures", fontsize=25, x=0.42);
    plt.tight_layout;


import sklearn


def matriz_confusion(y_real, y_pred, modelo):
    import matplotlib.pyplot as plt
    import scikitplot as skplt
    # Normalized confusion matrix for the K-NN
    skplt.metrics.plot_confusion_matrix(y_real, y_pred, normalize=True)
    plt.show()
    # matc=confusion_matrix(y_real, y_pred)

    # fig, ax = plot_confusion_matrix(conf_mat=matc,
    #                               colorbar=True)
    # fig.suptitle(modelo)
    # plt.show()
    plt.tight_layout()
    print(metrics.classification_report(y_real, y_pred, digits=4))
    return metrics.classification_report(y_real, y_pred, digits=4, output_dict=True)


def plot_trained(vgghist, title='', axs=None, exp_name=""):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(vgghist.history['loss'], label='train')
    ax1.plot(vgghist.history['val_loss'], label='val')
    # ax1.set_ylim(-0.1, 0.1)

    titulo1 = " loss"
    # ax1.set_title('loss')
    ax1.set_title(titulo1)

    ax1.legend()

    ax2.plot(vgghist.history['acc'], label='train accuracy')
    ax2.plot(vgghist.history['val_acc'], label='val accuracy')
    # ax2.set_ylim(0.9, 1.1)
    # ax2.set_title('Accuracy')
    titulo2 = " Accuracy"
    ax2.set_title(titulo2)

    ax2.legend()


# plot_trained(vgghist, title='Model_VGG', axs=None, exp_name="");


def plot_bucle(acc, val_acc, loss, val_loss, title='', axs=None, exp_name=""):
    f, axs = plt.subplots(2, figsize=(15, 10))

    axs[0].plot(loss, label='Training Set', )
    axs[0].plot(val_loss, label='Validation Set')
    # ax1.set_ylim(-0.1, 0.1)

    titulo1 = "Loss"
    # ax1.set_title('loss')
    axs[0].set_title('Model Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel(titulo1)
    axs[0].legend()

    axs[1].plot(acc, label='Training Set')
    axs[1].plot(val_acc, label='Validation Set')
    # ax2.set_ylim(0.9, 1.1)
    # ax2.set_title('Accuracy')

    titulo2 = "Accuracy"
    axs[1].set_title('Model Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel(titulo2)
    axs[1].legend()


def plot_test(acc_test, loss_test, title='', axs=None, exp_name=""):
    f, axs = plt.subplots(2, figsize=(15, 10))

    axs[0].plot(loss_test, label='Loss')
    titulo3 = "Loss"
    axs[0].set_title('Loss Test')
    axs[0].legend()
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel(titulo3)

    axs[1].plot(acc_test, label='Accuracy')
    titulo4 = "Accuracy"
    axs[1].set_title('Accuracy Test')
    axs[1].legend()
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel(titulo4)


# DataAugmentation

def DataAug(X_train, Y_train):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=15,
        zoom_range=[0.7, 1.4],
        horizontal_flip=True,
        vertical_flip=True
    )

    # plt.figure(figsize=(20,8))
    # for imagen, etiqueta in datagen.flow(X_train, Y_train, batch_size=10, shuffle=False):
    #   # for i in range(10):
    #   #   plt.subplot(2, 5, i+1)
    #   #   plt.xticks([])
    #   #   plt.yticks([])
    #   #   plt.imshow(imagen[i], cmap="gray")
    #   break

    # datagen.flow(X_train, Y_train, batch_size=32, shuffle=False)

    return (datagen, X_train, Y_train)


def create_model_vgg():
    gc.collect()

    base_model = VGG16(input_shape=(200, 200, 3),  # Shape of our images
                       include_top=False,  # Leave out the last fully connected layer
                       weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(base_model.output)

    x = layers.Dense(1024, activation='relu')(x)

    x = layers.Dense(512, activation='relu')(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)

    # x = layers.Dropout(0.5)(x)

    #  x = layers.Dense(128,activation = 'relu')(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)

    # model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
    # model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = 'binary_crossentropy',metrics = ['acc'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['acc'])

    tf.keras.utils.plot_model(model, to_file="my_model_vgg.png", show_shapes=True)

    return model


categories = ["NORMAL", "PNEUMONIA"]
datasets = ["train", "test", "val"]

data_vgg = []
target_vgg = []

from pathlib import Path, PureWindowsPath

# I've explicitly declared my path as being in Windows format, so I can use forward slashes in it.


for set_ in datasets:
    for cat in categories:
        filelist = PureWindowsPath("C:\\Users\\migue\\Desktop\\TFG\\TFG\\Dataset_y_codigos\\dataset\\chest_xray" + set_ + cat + "*.jpeg")
        correct_path = Path(filelist)
        print(correct_path)
      #  filelist = glob.glob( + set_ + '/' + cat + '/*.jpeg')
        target_vgg.extend([cat for _ in filelist])

        # data_vgg.extend([np.array(Image.open(fname).convert('L').resize((200, 200))) for fname in filelist])
        data_vgg.extend([np.array(Image.open(fname).convert('RGB').resize((200, 200))) for fname in filelist])
#
data_array_vgg = np.stack(data_vgg, axis=0)

# Nos quedamos con el 80% del dataset porque la memoria RAM de Google Colab revienta con un porcentaje mayor

i = data_array_vgg.shape[0] * 0.8

data_array_vgg = data_array_vgg[:int(i)]
target_vgg = target_vgg[:int(i)]

X_train_vgg, X_test_vgg, y_train_vgg, y_test_vgg = train_test_split(data_array_vgg, np.array(target_vgg),
                                                                    random_state=43, test_size=0.2, stratify=target_vgg)

X_test_vgg_norm = np.round((X_test_vgg / 255), 3).copy()
X_train_vgg_norm = np.round((X_train_vgg / 255), 3).copy()

encoder = LabelEncoder().fit(y_train_vgg)

y_train_vgg_cat = encoder.transform(y_train_vgg)
y_test_vgg_cat = encoder.transform(y_test_vgg)

del data_vgg
del y_train_vgg
del y_test_vgg
del X_test_vgg
del X_train_vgg

# Le hacemos reshape al dataset para poder utilizarlo en VGG16

gc.collect()

# X_train_vgg_rgb = X_train_vgg_norm.reshape(-1,200,200,1)
# X_test_vgg_rgb = X_test_vgg_norm.reshape(-1,200,200,1)

# X_train_vgg_rgb = np.repeat(X_train_vgg_rgb, 3, 3)
# X_test_vgg_rgb = np.repeat(X_test_vgg_rgb, 3, 3)


X_train_vgg_rgb = X_train_vgg_norm.reshape(-1, 200, 200, 3)
X_test_vgg_rgb = X_test_vgg_norm.reshape(-1, 200, 200, 3)

del X_train_vgg_norm
del X_test_vgg_norm

len(X_test_vgg_rgb) / 2

X_val_vgg_rgb, y_val_vgg_cat = X_test_vgg_rgb[:int(len(X_test_vgg_rgb) / 2)], y_test_vgg_cat[
                                                                              :int(len(y_test_vgg_cat) / 2)]
X_test_vgg_rgb, y_test_vgg_cat = X_test_vgg_rgb[int(len(X_test_vgg_rgb) / 2):], y_test_vgg_cat[
                                                                                int(len(y_test_vgg_cat) / 2):]

print(len(X_val_vgg_rgb))
print(len(y_val_vgg_cat))
print(len(X_test_vgg_rgb))
print(len(y_test_vgg_cat))

es = EarlyStopping(patience=5, monitor='val_acc', restore_best_weights=True)

i = 0
acc_vgg = []
val_acc_vgg = []
loss_vgg = []
val_loss_vgg = []
loss_vgg_test = []
acc_vgg_test = []
predictions_vgg = []
precision_0_vgg_test = []
recall_0_vgg_test = []
f1_0_vgg_test = []
precision_1_vgg_test = []
recall_1_vgg_test = []
f1_1_vgg_test = []

for i in range(10):
    model = create_model_vgg()

    print("Iteracion: " + str(i + 1))

    vgg_train = model.fit(X_train_vgg_rgb, y_train_vgg_cat, epochs=100, callbacks=es,
                          validation_data=(X_val_vgg_rgb, y_val_vgg_cat))
    # vgg_train = model.fit(X_train_vgg_rgb,y_train_vgg_cat, epochs = 35,validation_split=0.3)

    acc_vgg.append(vgg_train.history['acc'])
    val_acc_vgg.append((vgg_train.history['val_acc']))
    loss_vgg.append((vgg_train.history['loss']))
    val_loss_vgg.append((vgg_train.history['val_loss']))

    (loss_vtest, acc_vtest) = model.evaluate(X_test_vgg_rgb, y_test_vgg_cat)

    predictions_vgg.append(model.predict(X_test_vgg_rgb))
    variables_matriz = matriz_confusion(y_test_vgg_cat, (np.round(predictions_vgg[i])), "Modelo_VGG" + str(i + 1))
    precision_0_vgg_test.append(variables_matriz["0"]["precision"])
    precision_1_vgg_test.append(variables_matriz["1"]["precision"])
    recall_0_vgg_test.append(variables_matriz["0"]["recall"])
    recall_1_vgg_test.append(variables_matriz["1"]["recall"])
    f1_0_vgg_test.append(variables_matriz["0"]["f1-score"])
    f1_1_vgg_test.append(variables_matriz["1"]["f1-score"])

    print("\n")

    print("Test:" + "[" + str(loss_vtest) + "," + str(acc_vtest) + "]")

    loss_vgg_test.append(loss_vtest)
    acc_vgg_test.append(acc_vtest)

    variable_name = "vgg-16_" + str(i + 1) + ".h5"
    model.save('content/vgg/' + variable_name)

    gc.collect()

pos_max_vgg = acc_vgg_test.index(max(acc_vgg_test))
acc_vgg_ex = acc_vgg[pos_max_vgg]
val_acc_vgg_ex = val_acc_vgg[pos_max_vgg]
loss_vgg_ex = loss_vgg[pos_max_vgg]
val_loss_vgg_ex = val_loss_vgg[pos_max_vgg]

# precision_0_vgg_test_ex = precision_0_vgg_test[pos_max_vgg]
# recall_0_vgg_test_ex = recall_0_vgg_test[pos_max_vgg]
# f1_0_vgg_test_ex = f1_0_vgg_test[pos_max_vgg]
# precision_1_vgg_test_ex = precision_1_vgg_test[pos_max_vgg]
# recall_1_vgg_test_ex = recall_1_vgg_test[pos_max_vgg]
# f1_1_vgg_test_ex = f1_1_vgg_test[pos_max_vgg]


# vgg_test = model.fit(X_test_vgg_rgb,y_test_vgg_cat,epochs = 100,callbacks = es,validation_split=0.3)
# El mejor valor lo he obtenido con este modelo


gc.collect()

acc_vgg = np.array(acc_vgg)
val_acc_vgg = np.array(val_acc_vgg)
loss_vgg = np.array(loss_vgg)
val_loss_vgg = np.array(val_loss_vgg)
loss_vgg_test = np.array(loss_vgg_test)
acc_vgg_test = np.array(acc_vgg_test)
precision_0_vgg_test = np.array(precision_0_vgg_test)
recall_0_vgg_test = np.array(recall_0_vgg_test)
f1_0_vgg_test = np.array(f1_0_vgg_test)
precision_1_vgg_test = np.array(precision_1_vgg_test)
recall_1_vgg_test = np.array(recall_1_vgg_test)
f1_1_vgg_test = np.array(f1_1_vgg_test)

acc_vgg = np.concatenate(acc_vgg, axis=0)
val_acc_vgg = np.concatenate(val_acc_vgg, axis=0)
loss_vgg = np.concatenate(loss_vgg, axis=0)
val_loss_vgg = np.concatenate(val_loss_vgg, axis=0)

plot_bucle(acc_vgg, val_acc_vgg, loss_vgg, val_loss_vgg, title='Model_VGG', axs=None, exp_name="")

plot_test(acc_vgg_test, loss_vgg_test, title='Model_VGG_test', axs=None, exp_name="")

plot_bucle(acc_vgg_ex, val_acc_vgg_ex, loss_vgg_ex, val_loss_vgg_ex, title='Model_VGG_ex', axs=None, exp_name="")

desv_acc_vgg = acc_vgg.std()
desv_val_acc_vgg = val_acc_vgg.std()
desv_loss_vgg = loss_vgg.std()
desv_val_loss_vgg = val_loss_vgg.std()
desv_loss_vgg_test = loss_vgg_test.std()
desv_acc_vgg_test = acc_vgg_test.std()
desv_precision_0_vgg_test = precision_0_vgg_test.std()
desv_precision_1_vgg_test = precision_1_vgg_test.std()
desv_recall_0_vgg_test = recall_0_vgg_test.std()
desv_recall_1_vgg_test = recall_1_vgg_test.std()
desv_f1_0_vgg_test = f1_0_vgg_test.std()
desv_f1_1_vgg_test = f1_1_vgg_test.std()

print("\n")
print("Media Train_Acc: " + str(acc_vgg.mean().round(6)) + " " + u"\u00B1" + " " + str(desv_acc_vgg.mean().round(6)))
print("Media Val_Acc: " + str(val_acc_vgg.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_val_acc_vgg.mean().round(6)))
print("Media Train_Loss: " + str(loss_vgg.mean().round(6)) + " " + u"\u00B1" + " " + str(desv_loss_vgg.mean().round(6)))
print("Media Val_loss: " + str(val_loss_vgg.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_val_loss_vgg.mean().round(6)))
print("Media Acc Test: " + str(acc_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_acc_vgg_test.mean().round(6)))
print("Media loss Test: " + str(loss_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_loss_vgg_test.mean().round(6)))
print("\n")

print("\nEn la iteracion " + str(pos_max_vgg + 1) + " se ha conseguido el mayor valor de test\n")
print("\nSe ha obtenido el siguiente resultado en la iteracion " + str(pos_max_vgg + 1) + ":")
print("Test--> [" + str(loss_vgg_test[pos_max_vgg]) + "," + str(acc_vgg_test[pos_max_vgg]) + "]")

print("Precision 0 Test: " + str(precision_0_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_precision_0_vgg_test.mean().round(6)))
print("Precision 1 Test: " + str(precision_1_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_precision_1_vgg_test.mean().round(6)))
print("Recall 0 Test: " + str(recall_0_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_recall_0_vgg_test.mean().round(6)))
print("Recall 1 Test: " + str(recall_1_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_recall_1_vgg_test.mean().round(6)))
print("F1-Score 0 Test: " + str(f1_0_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_f1_0_vgg_test.mean().round(6)))
print("F1-Score 1 Test: " + str(f1_1_vgg_test.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_f1_1_vgg_test.mean().round(6)))

precision_vgg_total = np.append(precision_0_vgg_test, precision_1_vgg_test)
recall_vgg_total = np.append(recall_0_vgg_test, recall_1_vgg_test)
f1_vgg_total = np.append(f1_0_vgg_test, f1_1_vgg_test)

desv_precision_vgg = precision_vgg_total.std()
desv_recall_vgg = recall_vgg_total.std()
desv_f1_vgg = f1_vgg_total.std()

print("Precision Total Test: " + str(precision_vgg_total.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_precision_vgg.mean().round(6)))
print("Recall Total Test: " + str(recall_vgg_total.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_recall_vgg.mean().round(6)))
print("F1-Score Total Test: " + str(f1_vgg_total.mean().round(6)) + " " + u"\u00B1" + " " + str(
    desv_f1_vgg.mean().round(6)))

acc_vgg_ex_np = np.array(acc_vgg_ex)
val_acc_vgg_ex_np = np.array(val_acc_vgg_ex)
loss_vgg_ex_np = np.array(loss_vgg_ex)
val_loss_vgg_ex_np = np.array(val_loss_vgg_ex)

desv_acc_vgg_ex_np = acc_vgg_ex_np.std()
desv_val_acc_vgg_ex_np = val_acc_vgg_ex_np.std()
desv_loss_vgg_ex_np = loss_vgg_ex_np.std()
desv_val_loss_vgg_ex_np = val_loss_vgg_ex_np.std()

print("\n")
print("Media Train_Acc_VGG_iteracion_" + str(pos_max_vgg + 1) + ": " + str(
    acc_vgg_ex_np.mean().round(6)) + " " + u"\u00B1" + " " + str(desv_acc_vgg_ex_np.mean().round(6)))
print("Media Val_Acc_VGG_iteracion_" + str(pos_max_vgg + 1) + ": " + str(
    val_acc_vgg_ex_np.mean().round(6)) + " " + u"\u00B1" + " " + str(desv_val_acc_vgg_ex_np.mean().round(6)))
print("Media Train_Loss_VGG_iteracion_" + str(pos_max_vgg + 1) + ": " + str(
    loss_vgg_ex_np.mean().round(6)) + " " + u"\u00B1" + " " + str(desv_loss_vgg_ex_np.mean().round(6)))
print("Media Val_loss_VGG_iteracion_" + str(pos_max_vgg + 1) + ": " + str(
    val_loss_vgg_ex_np.mean().round(6)) + " " + u"\u00B1" + " " + str(desv_val_loss_vgg_ex_np.mean().round(6)))
print("\n")

gc.collect()

m = Accuracy()
m.update_state(y_test_vgg_cat, np.round(predictions_vgg[pos_max_vgg]))
print(m.result().numpy())
print(acc_vgg_test[len(acc_vgg_test) - 1])

acc = accuracy_score(y_test_vgg_cat, np.round(predictions_vgg[pos_max_vgg]))
print("Accuracy: " + str(acc))

