from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from abc import abstractmethod
from tensorflow import keras, nn
import tensorflow as tf
import numpy as np


class KerasEstimator(BaseEstimator):
    @abstractmethod
    def fit(self, X, y): # pragma: no cover
        pass

    @abstractmethod
    def transform(self, X): # pragma: no cover
        pass

    def predict(self, X):
        return self.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class KerasClassifier(KerasEstimator, ClassifierMixin):
    @abstractmethod
    def _init_model(self, X, y): # pragma: no cover
        pass

    def fit(self, X, y):
        self._init_model(X, y)

        X_shape = np.reshape(X, (-1,) + self.input_shape)
        y_cat = tf.keras.utils.to_categorical(y, 10)

        x_train, x_valid, y_train, y_valid = train_test_split(
            X_shape, y_cat, test_size=0.25, random_state=42)

        history = self.network.fit(x_train, y_train,
                                   epochs=self.max_iter,
                                   batch_size=self.batch_size,
                                   validation_data=(x_valid, y_valid),
                                   verbose=self.verbose)

        self.is_fitted_ = True
        return self

    def validate_inputs(self, X, y):
        return (X, y)

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        X_shape = np.reshape(X, (-1,) + self.input_shape)

        predictions = self.network.predict(X_shape, verbose=0)
        return predictions.argmax(axis=1)

    def transform(self, X):
        return self.predict(X)


class KerasCNNClassifier(KerasClassifier):
    def __init__(
        self,
        hidden_layer_sizes=(32,64),
        batch_size=8,
        input_shape=(28,28,1),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        alpha=0.0001,
        weight_decay=None,
        max_iter=10,
        verbose=0
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.verbose = verbose

        self.network = None

    def _init_model(self, X, y):
        self.network = keras.models.Sequential()

        is_first = True
        for layer_size in self.hidden_layer_sizes:
            if is_first:
                cnn = keras.layers.Conv2D(layer_size, (3,3),
                                          padding='same',
                                          input_shape=self.input_shape)
                is_first = False
            else:
                cnn = keras.layers.Conv2D(layer_size, (3,3),
                                          padding='same',
                                          activation=self.activation)
            self.network.add(cnn)
            self.network.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.network.add(keras.layers.Flatten())
        self.network.add(keras.layers.Dense(128, activation=self.activation, kernel_regularizer=keras.regularizers.L2(l2=self.alpha)))

        self.network.add(keras.layers.Dense(10, activation=nn.softmax, kernel_regularizer=keras.regularizers.L2(l2=self.alpha)))

        if self.solver == "adam":
            self.solver = keras.optimizers.Adam(learning_rate=self.learning_rate_init,
                                               beta_1=0.9,
                                               beta_2=0.999,
                                               weight_decay=self.weight_decay,
                                               amsgrad=False)
        if self.solver == "sgd":
            self.solver = keras.optimizers.SGD(learning_rate=self.learning_rate_init,
                                               weight_decay=self.weight_decay)

        self.network.compile(loss=keras.losses.categorical_crossentropy,
                             optimizer=self.solver, metrics=['categorical_crossentropy'])
