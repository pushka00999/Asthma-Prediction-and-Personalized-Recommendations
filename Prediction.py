import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, LSTM, Add, LayerNormalization, \
    ReLU, Multiply, Layer, Flatten, SimpleRNN
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, LayerNormalization, GRU, Add
import numpy as np
from sklearn import metrics
from math import sqrt
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import d2_pinball_score, explained_variance_score, max_error, mean_gamma_deviance, mean_pinball_loss, median_absolute_error, root_mean_squared_error, root_mean_squared_log_error


def error(actual, pred):
    err = np.empty(9)
    err[0] = metrics.mean_squared_error(actual, pred)
    err[1] = metrics.mean_absolute_error(actual, pred)
    err[2] = metrics.mean_squared_log_error(actual, pred)
    err[3] = root_mean_squared_error(actual, pred)
    err[4] = metrics.mean_absolute_percentage_error(actual, pred)
    err[5] = d2_pinball_score(actual, pred)
    err[6] = explained_variance_score(actual, pred)
    err[7] = mean_pinball_loss(actual, pred)
    err[8] = root_mean_squared_log_error(actual, pred)
    return err


class RMSLE(Loss):
    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Ensure predictions are not zero to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

        min_y_true = tf.reduce_min(y_true)
        min_y_pred = tf.reduce_min(y_pred)

        log_true = tf.math.log1p(y_true - min_y_true+1)  # improved rmsle
        log_pred = tf.math.log1p(y_pred - min_y_pred+1)

        return tf.sqrt(tf.reduce_mean(tf.square(log_true - log_pred)))

# Define multi-scale convolutional layers
def multi_scale_conv(inputs):
    conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    conv2 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
    conv3 = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(inputs)
    return Add()([conv1, conv2, conv3])


# Define a residual block
def residual_block(x, kernel_size=3, strides=1):
    res = Conv1D(32, kernel_size, padding='same', strides=strides)(x)
    res = BatchNormalization()(res)
    res = ReLU()(res)
    res = Conv1D(64, kernel_size, padding='same', strides=strides)(res)
    res = BatchNormalization()(res)
    res = Add()([res, x])
    return ReLU()(res)


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(input_shape[-1], activation='softmax')
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        attention_weights = self.dense(inputs)
        attention_output = Multiply()([inputs, attention_weights])
        return attention_output

    def compute_output_shape(self, input_shape):
        return input_shape


# Define the hybrid dilated convolution
def hybrid_dilated_conv(inputs):
    conv1 = Conv1D(filters=32, kernel_size=3, padding='same', dilation_rate=1, activation='relu')(inputs)
    conv2 = Conv1D(filters=32, kernel_size=3, padding='same', dilation_rate=2, activation='relu')(inputs)
    conv3 = Conv1D(filters=32, kernel_size=3, padding='same', dilation_rate=4, activation='relu')(inputs)
    return Add()([conv1, conv2, conv3])


def AsthmaNet(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 20)
    y_test = tf.keras.utils.to_categorical(y_test, 20)

    input_shape = x_train[1].shape

    inputs = Input(shape=input_shape)

    # Multi-scale convolutional layers
    x = multi_scale_conv(inputs)

    # Residual connections
    x = residual_block(x)

    # Attention gates
    x = Attention()(x)

    # Hybrid dilated convolution
    x = hybrid_dilated_conv(x)

    # RNN (LSTM) layer
    x = LSTM(64, return_sequences=True)(x)

    # Transformer layer (Simplified for illustration)
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = LayerNormalization()(x)

    x = Flatten()(x)

    # Output layer
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(20)(x)

    model = Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=RMSLE(),
                  metrics=[MeanSquaredError()])

    model.fit(x_train, y_train, batch_size=10, epochs=200, validation_data=(x_test, y_test), steps_per_epoch=25000, verbose=0)

    pred = abs(model.predict(x_test))

    return pred, error(y_test, pred), model


def Lstm(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 5)
    y_test = tf.keras.utils.to_categorical(y_test, 5)

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(32, input_shape=(x_train[1].shape)))
    model.add(Dense(5))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    model.fit(x_train, y_train, batch_size=10, epochs=200, validation_data=(x_test, y_test), steps_per_epoch=25000, verbose=0)

    predictions = abs(model.predict(x_test))

    return predictions, error(y_test, predictions)


def RNN(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 5)
    y_test = tf.keras.utils.to_categorical(y_test, 5)

    model = Sequential()
    model.add(SimpleRNN(64, input_shape=x_train[1].shape))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='mse', metrics=['mse'], optimizer='adam')
    model.fit(x_train, y_train, batch_size=10, epochs=200, validation_data=(x_test, y_test), steps_per_epoch=25000, verbose=0)
    pred = abs(model.predict(x_test))
    met = error(y_test, pred)
    return pred, met


def DNN(x_train, y_train, x_test, y_test):

    y_train = tf.keras.utils.to_categorical(y_train, 5)
    y_test = tf.keras.utils.to_categorical(y_test, 5)

    model = Sequential()
    model.add(Dense(128, input_shape=x_train[1].shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='mse', metrics=['mse'], optimizer='adam')
    model.fit(x_train, y_train, batch_size=10, epochs=200, validation_data=(x_test, y_test), steps_per_epoch=25000, verbose=0)
    pred = abs(model.predict(x_test))
    return pred, error(y_test, pred)


def ANN(x_train, y_train, x_test, y_test):

    y_train = tf.keras.utils.to_categorical(y_train, 5)
    y_test = tf.keras.utils.to_categorical(y_test, 5)

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=5))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    model.fit(x_train, y_train, batch_size=10, epochs=200, validation_data=(x_test, y_test), steps_per_epoch=25000, verbose=0)

    y_pred= abs(model.predict(x_test))

    return y_pred, error(y_test, y_pred)

