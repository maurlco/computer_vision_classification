import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 as cv
from keras import backend as K
import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout, UpSampling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD


class Model():

    def __init__(self, img_shape, name, mode, class_names):
        self.name = name
        self.img_shape = img_shape + (3,)
        self.mode = mode
        self.class_names = class_names

    def _recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def _precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def _f1_m(self, y_true, y_pred):
        precision = self._precision_m(y_true, y_pred)
        recall = self._recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    def _data_augmentation(self,inputs):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2)
        ])
        return data_augmentation(inputs)

    def add_aug_layer(self, layer):
        self._data_augmentation().add(layer)


    def _preprocess_input(self, name, data):
        if name == 'MobileNetV2':
            return tf.keras.applications.mobilenet_v2.preprocess_input(data)
        if name == 'ResNet50':
            return tf.keras.applications.resnet50.preprocess_input(data)
        else:
            print(f"The model {name} is not yet registered in this class")


    def _load_model(self):
        if self.name == 'MobileNetV2':
            return tf.keras.applications.MobileNetV2(input_shape=self.img_shape, include_top=False, weights='imagenet')
        if self.name == 'ResNet50':
            return tf.keras.applications.ResNet50(input_shape=self.img_shape, include_top=False, weights='imagenet')
        else:
            print(f'The model {self.name} you trying to load is not registered')

    def _build(self, mode):
        base_model = self._load_model()
        base_model.trainable = mode
        inputs = Input(shape=self.img_shape)
        x1 = self._data_augmentation(inputs)
        x2 = self._preprocess_input(self.name,x1)
        base_model_layer = base_model(x2, training=False)
        pooling_layer = GlobalAveragePooling2D()(base_model_layer)
        dropout_layer = Dropout(0.1)(pooling_layer)
        Layer_1 = Dense(512, activation='relu')(dropout_layer)
        outputs = Dense(12, activation='softmax')(Layer_1)
        model = tf.keras.Model(inputs, outputs)


        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy', self._f1_m])

        model.load_weights('models/CNN_mobileNetV2_model2.hdf5')

        return model


    def predict_breed(self, img_path):
        print('Dog to be identified : ')
        display(Image(img_path))
        print('Predicting the dog breed ... ')
        img_test = Image.open(img_path)
        img_array = np.array(img_test)
        res_img = cv.resize(img_array, (224,224), interpolation=cv.INTER_LINEAR)
        image = np.expand_dims(res_img, axis=0)
        mobilenet_model = self.build(self.mode)
        result = mobilenet_model.predict(image)
        print('Dog breed : ')
        return self.class_names[np.argmax(result)]