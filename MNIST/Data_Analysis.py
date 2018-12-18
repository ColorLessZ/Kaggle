import os
import sys
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def data_analysis_train_data_distribution(file_path):
    sns.set(style='white', context='notebook', palette='deep')
    data = pd.read_csv(file_path)
    data_y = data["label"]
    del data
    sns.countplot(data_y)
    print(data_y.value_counts())    

def data_analysis_null_check(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    train_x = train.drop(labels = ["label"],axis = 1)
    print(train_x.isnull().any().describe())
    print(test.isnull().any().describe())

if __name__ == "__main__":
    os.chdir(sys.path[0])
    train_file = "D:\\DeepLearning\\Kaggle\\MNIST\\data\\Original\\train.csv"
    test_file = "D:\\DeepLearning\\Kaggle\\MNIST\\data\\Original\\test.csv"
    data_analysis_train_data_distribution(train_file)
    data_analysis_null_check(train_file, test_file)