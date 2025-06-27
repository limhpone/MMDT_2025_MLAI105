import os
import numpy as np
import keras.applications

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
import tensorflow as tf
import numpy as np
from keras.utils import img_to_array  # type: ignore

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions    # type: ignore

from keras.applications.inception_v3 import preprocess_input as inception_preprocess
# from keras.applications.resnet import preprocess_input as resnet_preprocess
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.efficientnet import preprocess_input as effnet_preprocess

from keras.applications.convnext import preprocess_input as convnext_preprocess


class cnnModels:
    def __init__(self):
        self.models = {
            "ResNet50": self.resnet(),
            "VGGNet16": self.vggnet(),
            "InceptionV3": self.inception(),
            "ConvNeXt": self.convnet(),
            "EfficientNet": self.efficientnet(),
        }

    def resnet(self):
        model = keras.applications.ResNet50(weights="imagenet")
        return model

    def vggnet(self):
        model = keras.applications.VGG16(weights="imagenet")
        return model

    def inception(self):
        model = keras.applications.InceptionV3(weights="imagenet")
        return model

    def convnet(self):
        model = keras.applications.ConvNeXtTiny(weights="imagenet")
        return model

    def efficientnet(self):
        model = keras.applications.EfficientNetB7(weights="imagenet")
        return model

    def get_model(self, name):
        if name in self.models:
            return self.models[name]
        else:
            raise ValueError(f"Model '{name}' does not exist.")

    # def classify_image(self, name, img):
    #     model = self.get_model(name)
    #     img = img.resize((model.input_shape[1], model.input_shape[2]))
    #     x = img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x)
    #     preds = model.predict(x, verbose=0)
    #     return decode_predictions(preds, top=1)

    def classify_image(self, name, img, top_k=1):
        model = self.get_model(name)
        img = img.resize((model.input_shape[1], model.input_shape[2]))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Use correct preprocessing
        if name == "InceptionV3":
            x = inception_preprocess(x)
        elif name == "ResNet50":
            x = preprocess_input(x)
        elif name == "VGGNet16":
            x = vgg_preprocess(x)
        elif name == "EfficientNet":
            x = effnet_preprocess(x)
        elif name == "ConvNeXt":
            x = convnext_preprocess(x)
        else:
            x = preprocess_input(x)

        preds = model.predict(x, verbose=0) # type: ignore
        
        if name == "ConvNeXt": 
            preds = tf.nn.softmax(preds).numpy() # type: ignore
            
        return decode_predictions(preds, top=top_k)
