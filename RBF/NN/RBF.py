import numpy as np
import pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

#---------------------------------------

class InitCentersKMeans(Initializer):

    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


class InitCentersRandom(Initializer):

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):

        C = K.expand_dims(self.centers)
        H = K.transpose(C - K.transpose(x))
        return K.exp(-self.betas * K.sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#-----------------------
my_data = pd.read_excel('dataset.xls')

scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
my_data = scaler.fit_transform(my_data)

my_data = pd.DataFrame(my_data).to_numpy()

x = my_data[:, :4]
y = my_data[:, 4]
y = np.reshape(y, (-1, 1))


kfold_cv = KFold(n_splits=5, shuffle=True, random_state=13)
kfold_scores = []
kfold_r2 = []


model = Sequential()
model.add(RBFLayer(300, betas=0.5, input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))


Sthe = optimizers.Adadelta(learning_rate=18, rho=0.95)
model.compile(loss='mean_squared_error',
              optimizer=Sthe)

for train, test in kfold_cv.split(x, y):
    es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=50)
    history = model.fit(x[train], y[train],
                        epochs=100,
                        verbose=0,
                        batch_size=32,
                        callbacks=[])


#------------------
    scores = model.evaluate(x[test], y[test], verbose=0)
    r2 = r2_score(y[test], model.predict(x[test]))
    kfold_scores.append(scores)
    kfold_r2.append(r2)
    print("Fold Finished with Loss = %.5f & R^2 = %.3f" % (scores, r2))

print()
print('Results:')
print("Loss mean: %.5f " % (np.mean(kfold_scores)))
print("R^2 mean: %.5f " % (np.mean(kfold_r2)))


#----------------
yPredict = model.predict(x[test])
yTest = y[test]

plt.plot(yPredict, label='observed')
plt.plot(yTest, label='predicted')
#plt.plot(scalerX.inverse_transform(X_test))

plt.legend()

plt.show()