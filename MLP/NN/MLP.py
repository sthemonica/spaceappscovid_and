import pandas as pd

my_data = pd.read_excel('data/dataset.xls')

y = my_data [['casos']]
X = my_data[['densidade','idh','isolamento','abastecimento']]

#---------------------

from sklearn.preprocessing import MinMaxScaler

scalerX = MinMaxScaler()
scalerX.fit(X)
X = scalerX.transform(X)

scalery = MinMaxScaler()
scalery.fit(y)
y = scalery.transform(y)

#---------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#---------------------

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

mlp = MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(1500000,), activation='relu', max_iter=1000, alpha=1e-4,
                        solver='adam', verbose=10, random_state=1,
                    learning_rate_init=.1)).fit(X_train, y_train)

y_predicted = scalery.inverse_transform(mlp.predict(X_test))

#y_predicted = scalery.inverse_transform(y_predicted)

#---------------------

from sklearn.metrics import mean_squared_error, r2_score

yErrorMSE = mean_squared_error(scalery.inverse_transform(y_test), y_predicted)
yErrorR2 = r2_score(scalery.inverse_transform(y_test), y_predicted)

print(yErrorMSE)
print(yErrorR2)

#---------------------

import matplotlib.pyplot as plt

# Data for plotting
t = scalery.inverse_transform(y_test)
s = y_predicted

plt.plot(t, label='observed')
plt.plot(s, label='predicted')
#plt.plot(scalerX.inverse_transform(X_test))

plt.legend()

plt.show()