import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    os.chdir(sys.path[0])
    test_file = "D:\\DeepLearning\\Kaggle\\MNIST\\data\\Original\\test.csv"
    test = pd.read_csv(test_file)
    img = test.iloc[2].values.reshape(28,28)
    plt.imshow(img)
