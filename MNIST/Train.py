import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

if __name__ == "__main__":
    os.chdir(sys.path[0])

    #Load data
    train_file = "D:\\DeepLearning\\Kaggle\\MNIST\\data\\Original\\train.csv"
    train = pd.read_csv(train_file)
    train_y = train["label"]
    train_x = train.drop(labels = ["label"],axis = 1)

    #Normalize data
    train_x = train_x / 255.0

    #Reshape data
    train_x = train_x.values.reshape(-1, 28, 28, 1)

    #Label encoding
    train_y = to_categorical(train_y, num_classes = 10)

    #Split training and validation set
    random_seed = 2
    train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = random_seed)

    #Example display
    g = plt.imshow(train_x[0][:,:,0])

    #Define Model: [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (28, 28, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    #Define an optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    #Define a learning rate annealer (learning rate decay)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001)

    #Train model
    epochs = 200
    batch_size = 86

    #Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(train_x)

    #Model train without data augmentation
    #model.fit(train_x, train_y, batch_size = batch_size, epochs = epochs, validation_data = (validation_x, validation_y), verbose = 2)

    #Model train with data augmentation
    model.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size), epochs = epochs, validation_data = (validation_x, validation_y), verbose = 2, steps_per_epoch=train_x.shape[0] // batch_size, callbacks=[learning_rate_reduction])

    #Save model
    model_folder = ".\\Models\\"
    model_json = model.to_json()
    with open(model_folder + "model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_folder + "model.h5")
    print("Saved model to disk")


