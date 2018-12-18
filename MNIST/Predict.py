import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

if __name__ == "__main__":
    os.chdir(sys.path[0])
    model_folder = ".\\Models\\"
    test_file = "D:\\DeepLearning\\Kaggle\\MNIST\\data\\Original\\test.csv"
    prediction_file = "D:\\DeepLearning\\Kaggle\\MNIST\\data\\Prediction\\prediction_with_data_aug_4.csv"
    test = pd.read_csv(test_file)

    #Normalize data
    test = test / 255.0

    #Reshape data
    test = test.values.reshape(-1, 28, 28, 1)

    #Load model
    # load json and create model
    json_file = open(model_folder + "model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_folder + "model.h5")
    print("Loaded model from disk")
    
    predict_vec = loaded_model.predict(test, batch_size=None, verbose=0, steps=None)
    predict_classes = np.argmax(predict_vec,axis = 1) 

    out = [str(i+1) + "," + str(p) for i, p in enumerate(predict_classes)]
    with open(prediction_file, "w+") as f:
        f.writelines("ImageId,Label\n"+"\n".join(out))

