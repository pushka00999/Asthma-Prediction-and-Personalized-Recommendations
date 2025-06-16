'''
pip install tensorflow
pip install pandas
pip install statsmodels
pip install scikit-learn
pip install matplotlib
pip install openpyxl
pip install seaborn
'''

import os

os.makedirs('Saved Data', exist_ok=True)
os.makedirs('Results', exist_ok=True)
os.makedirs('Data Visualization', exist_ok=True)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from datagen import datagen
from save_load import load, save
from Prediction import AsthmaNet, Lstm, ANN, RNN, DNN
from plot_result import plotres
import matplotlib.pyplot as plt


def main():
    datagen()

    # 70 training, 30 testing

    x_train_70 = load('x_train_70')
    x_test_70 = load('x_test_70')
    y_train_70 = load('y_train_70')
    y_test_70 = load('y_test_70')

    # 80 training, 20 testing

    x_train_80 = load('x_train_80')
    x_test_80 = load('x_test_80')
    y_train_80 = load('y_train_80')
    y_test_80 = load('y_test_80')

    training_data = [(x_train_70, y_train_70, x_test_70, y_test_70), (x_train_80, y_train_80, x_test_80, y_test_80)]

    i = 70

    for train_data in training_data:
        x_train, y_train, x_test, y_test = train_data

        pred, met, model = AsthmaNet(x_train, y_train, x_test, y_test)

        save(f'proposed_{i}', met)
        save(f'predicted_{i}', pred)

        pred, met = ANN(x_train, y_train, x_test, y_test)
        save(f'ann_{i}', met)

        pred, met = DNN(x_train, y_train, x_test, y_test)
        save(f'dnn_{i}', met)

        pred, met = RNN(x_train, y_train, x_test, y_test)
        save(f'rnn_{i}', met)

        pred, met = Lstm(x_train, y_train, x_test, y_test)
        save(f'lstm_{i}', met)

        i += 10


a = 0
if a == 1:
    main()

plotres()
plt.show()

