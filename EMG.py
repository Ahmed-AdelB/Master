import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numpy import array
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_pickle("emg_dataset_pandas_dataframe.pkl")

scaler = preprocessing.StandardScaler()

arr = df[df.columns[:8]].values
arr = scaler.fit_transform(arr)

arr = np.reshape(arr, (-1, 58, 8))
arr = np.reshape(arr, (-1, 464))
labels = np.zeros(arr.shape[0], dtype=np.int64)
labels[5180:] = 1

unique, counts = np.unique(labels, return_counts=True)
x_train, x_test, y_train, y_test = train_test_split(
    arr, labels, shuffle="True", test_size=0.2
)

x_train = x_train.reshape((-1, 58, 8))
x_test = x_test.reshape((-1, 58, 8))
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# define model
model = Sequential()

model.add(
    Conv1D(
        filters=64,
        kernel_size=8,
        activation="relu",
        kernel_initializer="he_normal",
        input_shape=(58, 8),
    )
)
model.add(MaxPooling1D(pool_size=2))
model.add(
    Conv1D(
        filters=128, kernel_size=8, kernel_initializer="he_normal", activation="relu"
    )
)
model.add(MaxPooling1D(pool_size=2))
model.add(
    Conv1D(
        filters=128, kernel_size=8, kernel_initializer="he_normal", activation="relu"
    )
)
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dropout(rate=0.3))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(rate=0.3))
model.add(Dense(2, activation="softmax"))
adam = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["acc"])

# fit model


model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=100,
    verbose=1,
    validation_data=(x_test, y_test),
)
