from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

#  Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['Price'])

CRIM_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

# property_stats = np.ndarray(shape=(1, 11))
# property_stats[0][CRIM_IDX] = features['CRIM'].mean()
# property_stats[0][ZN_IDX] = features['ZN'].mean()
# property_stats[0][CHAS_IDX] = features['CHAS'].mean()

property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms, students_per_classroom):
    log_estimate = regr.predict(property_stats)
    return log_estimate

get_log_estimate()








