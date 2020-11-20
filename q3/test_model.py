import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def set_random_seed(seed_value=1004):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    
def loss_function(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def make_network(input_size):
    inputs = Input(shape=(input_size,), name='input_data')
    
    x = Dense(16, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(196, activation='relu')(x)    
    out = Dense(1, activation='linear', name='MEDV')(x)

    model = Model(inputs=inputs, outputs=out)
    # model.summary()
    return model

def make_housing_data(path):
    data_list = []
    f = open(path, 'r')
    while True:
        line = f.readline()
        if not line: break
        data_list.append([float(dd) for dd in line.split(' ') if dd != ''])
    f.close()
    return data_list

'''
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built prior to 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s
'''
if __name__ == '__main__':
    set_random_seed()
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    train_columns = ['RM', 'PTRATIO', 'CHAS', 'TAX', 'ZN', 'AGE', 'LSTAT']
    
    try:
        data = pd.read_csv('housing.csv')
    except:        
        data = pd.DataFrame(make_housing_data('housing.data'), columns=columns)
        data.to_csv('housing.csv', index=False)

    corr = data.corr()
    print(corr)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns_plot = sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', vmin=-1, vmax=1, cbar_kws={"shrink": .8})
    plt.title('Feature cross correlation matrix')
    sns_plot.get_figure().savefig('corr.png')

    normalized_data = data / data.max()

    index = np.arange(len(normalized_data))
    np.random.shuffle(index)

    split_rate = 0.8

    max_MEDV = data['MEDV'].max()
    
    X_train_data = normalized_data[train_columns].values[index[:int(len(index) * split_rate)]]
    y_train_data = normalized_data[columns[-1:]].values[index[:int(len(index) * split_rate)]]
    X_test_data = normalized_data[train_columns].values[index[int(len(index) * split_rate):]]
    y_test_data = normalized_data[columns[-1:]].values[index[int(len(index) * split_rate):]]

    

    result = np.zeros((len(y_test_data), 1))
    optimizer = ['mse', 'mae', loss_function]
    ensemble_num = 3
    for i in range(ensemble_num):
        model = make_network(len(train_columns))
        model.compile(optimizer='adam', loss=optimizer[i])
        # model.fit(X_train_data, y_train_data, batch_size=32, epochs=50, verbose=0)
        # model.save_weights('best_model_' + str(i) + '.h5')

        model.load_weights('best_model_' + str(i) + '.h5')
        result += model.predict(X_test_data)/ensemble_num
    
    print('Feature List :', train_columns)
    print('Result -------------')
    print(pd.DataFrame(np.concatenate([result * max_MEDV, y_test_data * max_MEDV], axis=1), columns=['predict', 'True']))
    print('RMSE : ', np.sqrt(np.mean(((y_test_data - result) * max_MEDV)**2)))
    print('R2 Score :', r2_score(y_test_data * max_MEDV, result * max_MEDV))
    plot_model(model, to_file='model_structure.png', show_shapes=True)