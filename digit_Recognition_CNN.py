from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import joblib


def input_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('int')
    X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('int')

    # normalize to range [0, 1]
    X_train = (X_train / 255)
    X_test = (X_test / 255)

    return X_test, y_test, X_train, y_train


def create_and_save_Model():
    X_test, y_test, X_train, y_train = input_data()
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    neigh.fit(X_train, y_train)
    joblib.dump(neigh, 'KNN.pkl')
    print("KNN Model saved to disk")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    # One hot Code
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

    # save model and architecture to single file
    model.save("cnn.hdf5")
    print("CNN Saved model to disk")
