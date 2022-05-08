from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# gathering data
boston_dataset = load_boston()
dir(boston_dataset)
boston_dataset.DESCR
boston_dataset.data

# convert ndarray to Dataframe
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['Price'] = boston_dataset.target

first_5 = data.head()
last_5 = data.tail()
column_count = data.count()

# check for missing values
pd.isnull(data).any()
data.info()

# visualising data - plotting histogram distributions and bar charts (matplotlib)
plt.figure(figsize=(10, 6))
plt.hist(data['Price'], bins=30, ec='black', color='lime')
plt.xlabel('Price in Thousands')
plt.ylabel('No. of Houses')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(data['RM'], ec='black', color='lime')
plt.xlabel('Average no. of rooms')
plt.ylabel('No. of Houses')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(data['RAD'], bins=24, ec='black', color='lime', rwidth=1)
plt.xlabel('Accessibility to Highways')
plt.ylabel('No. of Houses')
plt.show()

frequency = data['RAD'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(frequency.index, height=frequency)
plt.xlabel('Accessibility to Highways')
plt.ylabel('No. of Houses')
plt.show()

# visualising data - plotting histogram distributions and bar charts (seaborn)
plt.figure(figsize=(10, 6))
sns.distplot(data['Price'], bins=50, )
plt.show()

# Descriptive statistics
minimum = data['RM'].min()
maximum = data['RM'].max()
mean = data['RM'].mean()
median = data['RM'].median()

All_minimum = data.min()
All_maximum = data.max()
All_mean = data.mean()
All_median = data.median()

data.describe()

# correlation
data['Price'].corr(data['RM'])
data['Price'].corr(data['PTRATIO'])

data.corr()

mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

# visualizing correlations with heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={'size': 14})
sns.set_style('white')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# visualize with scatter using matplot
nox_dis_corr = round(data['NOX'].corr(data['DIS']), 3)
plt.figure(figsize=(9, 6))
plt.scatter(x=data['DIS'], y=data['NOX'], alpha=0.6, s=80, color='indigo')
plt.xlabel('DIS - distance from employment', fontsize=14)
plt.ylabel('NOX = Nitric oxide pollution', fontsize=14)
plt.title(f'DIS vs NOX (Correlation {nox_dis_corr}')
plt.show()

# visualize with scatter using seaborn
sns.set()
sns.set_context('talk')
sns.set_style('whitegrid')
sns.jointplot(x=data['DIS'], y=data['NOX'], size=7, color='indigo', joint_kws={'alpha': 0.5})
plt.show()

sns.lmplot(x='TAX', y='RAD', data=data, size=7)
plt.show()

rm_tgt_corr = round(data['RM'].corr(data['Price']), 3)
plt.figure(figsize=(9, 6))
plt.scatter(x=data['RM'], y=data['Price'], alpha=0.6, s=80, color='indigo')
plt.xlabel('DIS - distance from employment', fontsize=14)
plt.ylabel('Price - Property price', fontsize=14)
plt.title(f'RM vs Price (Correlation {rm_tgt_corr}')
plt.show()

sns.lmplot(x='RM', y='Price', data=data, size=7)
plt.show()

sns.pairplot(data, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
plt.show()

# training and test dataset split
from sklearn.model_selection import train_test_split

prices = data["Price"]
features = data.drop('Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

# running multivariable regression
from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# getting coefficients of multiple linear regression equation
# calculating model fit with R-squared
print('training data R-squared:', regr.score(X_train, y_train))
print('testing data R-squared:', regr.score(X_test, y_test))
print('intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])

# Data transformation (into log)
data['Price'].skew()
y_log = np.log(data['Price'])

sns.distplot(y_log)
plt.title(f'f]Log price with skew {y_log.skew()}')
plt.show()

transformed_data = features
transformed_data['LOG_PRICE'] = y_log
sns.lmplot(x='LSTAT', y='LOG_PRICE', data=transformed_data, size=7, scatter_kws={'alpha': 0.6},
           line_kws={'color': 'darkred'})
plt.show()

# regression using logged transformed data
prices = np.log(data["Price"])
features = data.drop('Price', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('training data R-squared:', regr.score(X_train, y_train))
print('testing data R-squared:', regr.score(X_test, y_test))
print('intercept', regr.intercept_)
pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['coef'])

# p_values & evaluating coefficients ordinary least squares
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()
results.params
results.pvalues

pd.DataFrame({'coef': results.params, 'p_value': round(results.pvalues, 3)})

# Testing for multicollinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF = variance_inflation_factor(exog=X_incl_const.values, exog_idx=1)

length = len(X_incl_const.columns)
length_2 = X_incl_const.shape[1]

VIF_list = []
for i in range(length):
    VIF = variance_inflation_factor(exog=X_incl_const.values, exog_idx=i)
    VIF_list.append(VIF)
    print(VIF_list)

pd.DataFrame({'Coef_name': X_incl_const.columns, 'VIF': np.around(VIF_list, 2)})

# Model simplification and Baysian information criterion (BIC)
# original model
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

original_coef = pd.DataFrame({'Coef_name': X_incl_const.columns, 'VIF': np.around(VIF_list, 2)})

print('BIC=', results.bic)
print('r_squared =', results.rsquared)

# reduced model number 1 excluding INDUS
X_incl_const = sm.add_constant(X_train)
X_incl_const = X_incl_const.drop(['INDUS'], axis=1)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

length = len(X_incl_const.columns)

VIF_list = []
for i in range(length):
    VIF = variance_inflation_factor(exog=X_incl_const.values, exog_idx=i)
    VIF_list.append(VIF)
    print(VIF_list)

coef_minus_indus = pd.DataFrame({'Coef_name': X_incl_const.columns, 'VIF': np.around(VIF_list, 2)})

print('BIC=', results.bic)
print('r_squared =', results.rsquared)

# reduced model number 2 excluding INDUS and age
X_incl_const = sm.add_constant(X_train)
X_incl_const = X_incl_const.drop(['INDUS', 'AGE'], axis=1)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

length = len(X_incl_const.columns)

VIF_list = []
for i in range(length):
    VIF = variance_inflation_factor(exog=X_incl_const.values, exog_idx=i)
    VIF_list.append(VIF)
    print(VIF_list)

reduced_coef = pd.DataFrame({'Coef_name': X_incl_const.columns, 'VIF': np.around(VIF_list, 2)})

print('BIC=', results.bic)
print('r_squared =', results.rsquared)

frames = [original_coef, coef_minus_indus, reduced_coef]
pd.concat(frames, axis=1)

#  Residuals and residual plots
# modified model: transformed (using log prices) and simplified (dropping two features)

prices = np.log(data["Price"])
features = data.drop(['Price', 'INDUS', 'AGE'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

# using statsmodel
X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

# Residuals
residuals = y_train - results.fittedvalues
# or
residuals_2 = results.resid

# Graph of actual vs predicted prices
corr = round(y_train.corr(results.fittedvalues), 2)
print(corr)
plt.scatter(y_train, results.fittedvalues, c='navy', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual log proces $y_i$', fontsize=14)
plt.ylabel('Predicted log proces $\hat y_i$', fontsize=14)
plt.title(f'Actual vs Preicted log prices: $y_i$ vs $\hat y_i$ (Corr{corr})', fontsize=17)
plt.show()

plt.scatter(x=np.e ** y_train, y=np.e ** results.fittedvalues, c='blue', alpha=0.6)
plt.plot(np.e ** y_train, np.e ** y_train, color='cyan')
plt.xlabel('Actual prices 000s $y_i$', fontsize=14)
plt.ylabel('Predicted prices 000s $\hat y_i$', fontsize=14)
plt.title(f'Actual vs Preicted prices: $y_i$ vs $\hat y_i$ (Corr{corr})', fontsize=17)
plt.show()

# residuals vs predicted values

plt.scatter(x=results.fittedvalues, y=results.resid, c='navy', alpha=0.6)
plt.xlabel('Predicted log proces $\hat y_i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title(f'Residuals vs Fitted data', fontsize=17)
plt.show()

# distribution of residuals (log prices) - checking for normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color='navy')
plt.title(f'Log price model: residuals skew ({resid_skew}), residuals mean ({resid_mean}')
plt.show()

# Mean squared error and R squared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)

# Original model: normal prices and all features

prices = data["Price"]
features = data.drop(['Price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

# Graph of actual vs predicted prices
corr = round(y_train.corr(results.fittedvalues), 2)
print(corr)
plt.scatter(y_train, results.fittedvalues, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual prices $y_i$', fontsize=14)
plt.ylabel('Predicted prices $\hat y_i$', fontsize=14)
plt.title(f'Actual vs Preicted prices: $y_i$ vs $\hat y_i$ (Corr{corr})', fontsize=17)
plt.show()


# residuals vs predicted values

plt.scatter(x=results.fittedvalues, y=results.resid, c='indigo', alpha=0.6)
plt.xlabel('Predicted prices $\hat y_i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title(f'Residuals vs Fitted data', fontsize=17)
plt.show()

# distribution of residuals - checking for normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color='indigo')
plt.title(f'Residuals skew ({resid_skew}), residuals mean ({resid_mean}')
plt.show()

# Mean squared error and R squared
full_normal_mse = round(results.mse_resid, 3)
full_normal_rsquared = round(results.rsquared, 3)

# Model omitting key Features using log prices

prices = np.log(data["Price"])
features = data.drop(['Price', 'INDUS', 'AGE', 'LSTAT', 'RM', 'NOX', 'CRIM'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=10)

X_incl_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_incl_const)
results = model.fit()

# Graph of actual vs predicted prices
corr = round(y_train.corr(results.fittedvalues), 2)
print(corr)
plt.scatter(y_train, results.fittedvalues, c='red', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.xlabel('Actual log prices $y_i$', fontsize=14)
plt.ylabel('Predicted log prices $\hat y_i$', fontsize=14)
plt.title(f'Actual vs Preicted prices with omittd features: $y_i$ vs $\hat y_i$ (Corr{corr})', fontsize=17)
plt.show()


# residuals vs predicted values

plt.scatter(x=results.fittedvalues, y=results.resid, c='red', alpha=0.6)
plt.xlabel('Predicted prices $\hat y_i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title(f'Residuals vs Fitted data', fontsize=17)
plt.show()

# distribution of residuals - checking for normality
resid_mean = round(results.resid.mean(), 3)
resid_skew = round(results.resid.skew(), 3)

sns.distplot(results.resid, color='red')
plt.title(f'Residuals skew ({resid_skew}), residuals mean ({resid_mean}')
plt.show()

# Mean squared error and R squared
reduced_log_mse = round(results.mse_resid, 3)
reduced_log_rsquared = round(results.rsquared, 3)

# Mean squared error and R squared
omitted_var_mse = round(results.mse_resid, 3)
omitted_var_rsquared = round(results.rsquared, 3)


metrics = pd.DataFrame({'R-squared':[reduced_log_rsquared, omitted_var_rsquared, full_normal_rsquared],
              'MSE': [reduced_log_mse, omitted_var_mse, full_normal_mse],
              'RMSE': np.sqrt([reduced_log_mse, omitted_var_mse full_normal_mse])},
             index=['reduced log model', 'full normal price model', 'omitted variable model'])

print('1 s.d in log prices is:', np.sqrt(reduced_log_mse))
print('2 s.d in log prices is:', 2*np.sqrt(reduced_log_mse))

upper_bound = np.log(30) + 2*np.sqrt(reduced_log_mse)
print('The upper bound in log prices for a 95% prediction interval is:', upper_bound)
print('The upper bound in normal prices for a 95% prediction interval is:', np.e**upper_bound*1000)

lower_bound = np.log(30) - 2*np.sqrt(reduced_log_mse)
print('The lower bound in log prices for a 95% prediction interval is:', lower_bound)
print('The lower bound in normal prices for a 95% prediction interval is:', np.e**lower_bound*1000)































