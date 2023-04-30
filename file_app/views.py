from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json


class FileView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            # ---------------------------------------------------------------------------- #
            #           here you can call your model function that process image           #
            # ---------------------------------------------------------------------------- #
            # ---------------------------------------------------------------------------- #
            #              get the image path from file_serilizer.data['file']             #
            # ---------------------------------------------------------------------------- #

            dic_res = {}

            def load_model():
                return keras.models.load_model('assets/best_plant_model_adil.h5')
            
            def load_classes():
                with open('assets/classes.txt', 'r') as f:
                    class_names = f.read().splitlines()
                return class_names

            def load_prep(img_path):
                img = tf.io.read_file(img_path)
                img = tf.image.decode_image(img)
                img = tf.image.resize(img, size=(224, 224))
                return img
              
            def make_prediction(model, image, classes):
                pred = model.predict(tf.expand_dims(image, axis=0))
                pred_idx = np.argmax(pred)
                # pred_prob = pred[0, pred_idx]*100
                predicted_value = classes[pred_idx]
                # print(f'predicted class: {predicted_value} with probability: {pred_prob:.2f}')
                return predicted_value
            
            path = file_serializer.data['file']
            path_mqad = path[1:]
            
            image = load_prep(path_mqad)
            model = load_model()
            classes = load_classes()
            prediction = make_prediction(model=model, classes=classes, image=image)
            
            result = prediction.split(sep="___")
            dic_res["Name"] = result[0].title()
            dic_res["Condition"] = result[1].title()
            dic_res["Type"] = result[2].title()
            
            print(file_serializer.data['file'])
            return Response(dic_res, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
