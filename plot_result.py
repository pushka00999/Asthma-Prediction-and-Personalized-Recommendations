import matplotlib.pyplot as plt
import numpy as np
from save_load import load, save
import pandas as pd
from sklearn import metrics
import os
os.makedirs('Results', exist_ok=True)


def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learning Rate (%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learning Rate (%)',
            kind='bar',
            stacked=False)

    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.legend(loc='upper right', prop={'size': 10, 'weight': 'bold', 'family': 'serif'})
    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)



def plotres():

    # 80, 20 variation
    ann_80 = load('ann_80')
    dnn_80 = load('dnn_80')
    rnn_80 = load('rnn_80')
    lstm_80 = load('lstm_80')
    proposed_80 = load('proposed_80')

    data = {
        'ANN': ann_80,
        'DNN': dnn_80,
        'RNN': rnn_80,
        'LSTM': lstm_80,
        'PROPOSED': proposed_80
    }

    ind = ['MSE', 'MAE', 'MSLE', 'RMSE', 'MAPE', 'DPS', 'EVS', 'MPL', 'RMSLE']
    table = pd.DataFrame(data, index=ind)
    save('table1', table)
    tab = table.to_excel('./Results/table_80.xlsx')

    val1 = np.array(table)

    # learn rate 70, 30
    ann_70 = load('ann_70')
    dnn_70 = load('dnn_70')
    rnn_70 = load('rnn_70')
    lstm_70 = load('lstm_70')
    proposed_70 = load('proposed_70')

    data1 = {
        'ANN': ann_70,
        'DNN': dnn_70,
        'RNN': rnn_70,
        'LSTM': lstm_70,
        'PROPOSED': proposed_70
    }

    ind = ['MSE', 'MAE', 'MSLE', 'RMSE', 'MAPE', 'DPS', 'EVS', 'MPL', 'RMSLE']
    table1 = pd.DataFrame(data1, index=ind)
    save('table2', table1)
    tab = table1.to_excel('./Results/table_70.xlsx')

    val2 = np.array(table1)

    method = ["ANN", "DNN", "RNN", "LSTM", "PROPOSED"]
    metrices_plot = ['MSE', 'MAE', 'MSLE', 'RMSE', 'MAPE', 'DPS', 'EVS', 'MPL', 'RMSLE']
    metrices = [val2, val1]
    save('met', metrices)

    learn_rate = [70, 80]

    for i in range(len(metrices_plot)):
        bar_plot(method, metrices[0][i, :], metrices[1][i, :],
                 metrices_plot[i])

    j = 0
    for i in learn_rate:
        print('Metrices-learning rate--' + str(i))
        tab = pd.DataFrame(metrices[j], index=metrices_plot, columns=method)
        print(tab)
        j+=1



