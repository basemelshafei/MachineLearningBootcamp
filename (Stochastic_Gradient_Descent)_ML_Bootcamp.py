#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:06:26 2022
batch gradient descend
@author: basemelshafei
"""
#  imports and packages

import matplotlib.pyplot as plt
import numpy as np 


# # Exapmle 1 a simple cost fucntion
#
# def f(x):
#     return x**2 + x+1
#
# def df(x):
#     return  2*x + 1
#
# #  make data
#
# x_1 = np.linspace(start = -3 , stop= 3, num= 500)
#
# # plot the data and derivative side by side
#
#
#
#
# # superimpose the gradient descent calculations
# new_x = 3
# previous_x = 0
# step_multiplier = 0.1
# precision = 0.00001
#
# x_list= [new_x]
# slope_list = [df(new_x)]
#
# plt.figure(figsize = [15, 5])
#
# plt.subplot(1,2,1)
#
# plt.title("cost function", fontsize = 17 )
# plt.xlim([-3, 3])
# plt.ylim(0, 8)
# plt.xlabel("x", fontsize=16)
# plt.ylabel("f(x)", fontsize = 16)
#
# plt.plot(x_1, f(x_1), color = "blue", linewidth = 3, alpha = 0.8)
# values = np.array(x_list)
# plt.scatter(x_list, f(values), color="red", s= 100, alpha = 0.6 )
#
# # Slope and derivative
#
# plt.subplot(1,2,2)
#
# plt.title("slope of the cost function", fontsize = 17 )
# plt.xlim([-2, 3])
# plt.ylim(-3, 6)
# plt.grid()
# plt.xlabel("x", fontsize=16)
# plt.ylabel("df(x)", fontsize = 16)
#
# plt.plot(x_1, df(x_1), color = "skyblue", linewidth = 5, alpha = 0.6)
# plt.scatter(x_list, slope_list, color= "red", s= 100, alpha = 0.5)
#
# plt.show()
#
#
#
# for n in range (500):
#     previous_x = new_x
#     gradient = df(previous_x)
#     new_x = previous_x - step_multiplier * gradient
#
#     step_size = abs(new_x - previous_x)
#     # print(step_size)
#
#     x_list.append(new_x)
#     slope_list.append(df(new_x))
#
#     if step_size < precision :
#         print("loop ran this many times:", n)
#         break
#
# print("local minimum occcurs at:" , new_x)
# print("slope or df(x0 value at this point is :", df(new_x))
# print("f(x) value or cost at this point is:", f(new_x))

# function 2
# Gradient Descent as a python function:

x_2 = np.linspace(-2, 2, 1000)

def g(x):
    return x**4 -4*x**2 + 5

def dg(x):
    return 4*x**3 - 8*x

def gradient_descent(derivative_func, initial_guess, multiplier=0.02, precision=0.001, max_iter = 300):
    new_x = initial_guess
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range (max_iter):
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - multiplier * gradient
        step_size = abs(new_x - previous_x)
        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))

        if step_size < precision:
            break
    return new_x, x_list, slope_list

local_min, list_x, deriv_list = gradient_descent(derivative_func=dg, initial_guess=-0.2, max_iter=10)

print("local min is:", local_min)
print("number of steps:", len(list_x))

plt.figure(figsize=[15, 5])

plt.subplot(1, 2, 1)

plt.title("cost function", fontsize=17)
plt.xlim([-2, 2])
plt.ylim(0.5, 5.5)
plt.xlabel("x", fontsize=16)
plt.ylabel("g(x)", fontsize=16)

plt.plot(x_2, g(x_2), color = "blue", linewidth = 3, alpha=0.8)
plt.scatter(list_x, g(np.array(list_x)), color='red', s=100, alpha=0.6)

# Slope and derivative

plt.subplot(1,2,2)

plt.title("slope of the cost function", fontsize=17)
plt.xlim([-2, 2])
plt.ylim(-6, 8)
plt.grid()
plt.xlabel("x", fontsize=16)
plt.ylabel("dg(x)", fontsize=16)

plt.plot(x_2, dg(x_2), color="skyblue", linewidth=5, alpha = 0.6)
plt.scatter(list_x, deriv_list, color="red", s=100, alpha=0.5)

plt.show()

# example 3 divergence, overflow, and python tuples

x_3 = np.linspace(-2.5, 2.5, 1000)

def h(x):
    return x**5 - 2*x**4 +2

def dh(x):
    return 5*x**4 -8*x**3

local_min, list_x, deriv_list = gradient_descent(derivative_func=dh, initial_guess=-0.2, max_iter=70)

print("local min is:", local_min)
print("number of steps:", len(list_x))

plt.figure(figsize=[15, 5])

plt.subplot(1, 2, 1)

plt.title("cost function", fontsize=17)
plt.xlim([-1.2, 2.5])
plt.ylim(-1, 4)
plt.xlabel("x", fontsize=16)
plt.ylabel("h(x)", fontsize=16)

plt.plot(x_3, h(x_3), color="blue", linewidth=3, alpha=0.8)
plt.scatter(list_x, h(np.array(list_x)), color='red', s=100, alpha=0.6)

# Slope and derivative

plt.subplot(1, 2, 2)

plt.title("slope of the cost function", fontsize=17)
plt.xlim([-1, 2])
plt.ylim(-4, 5)
plt.grid()
plt.xlabel("x", fontsize=16)
plt.ylabel("dh(x)", fontsize=16)

plt.plot(x_3, dh(x_3), color="skyblue", linewidth=5, alpha=0.6)
plt.scatter(list_x, deriv_list, color="red", s=100, alpha=0.5)

plt.show()

print("local min occurs at:", local_min)
print("the cost at this minimum is:", h(local_min))
print("number of steps:", len(list_x))


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import symbols, diff

from mpl_toolkits.mplot3d.axes3d import Axes3D


# Data viz with 3D charts

def f(x, y):
    r = 3 ** (-x ** 2 - y ** 2)
    return 1 / (r + 1)


#  Make our x and y data

x = np.linspace(start=-2, stop=2, num=200)
y = np.linspace(start=-2, stop=2, num=200)

print("shape of x array", x.shape)
#  make x and y 2D

x, y = np.meshgrid(x, y)

# generating 3d plot

fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.set_zlabel("f(x, y) - cost", fontsize=20)
ax.plot_surface(x, y, f(x, y), cmap=cm.coolwarm, alpha=0.4)

plt.show()

#  Partial derivatives and symbolic computations

a, b = symbols("x, y")

print("cost function is: ", f(a, b))
print("OUr derivative wrt x is: ", diff(f(a, b), a))
print("value of f(x,y) at x = 1.8 and y=1.0 is:", f(a, b).evalf(subs={a: 1.8, b: 1.0}))
print("value of partial derivative wrt x:", diff(f(a, b), a).evalf(subs={a: 1.8, b: 1.0}))

# batch gradient descent with sympy
#
# multiplier = 0.1
# max_iter = 200
# params = np.array([1.8, 1.0])  # initial guess
#
#
# for n in range(max_iter):
#     gradient_x = diff(f(a, b), a).evalf(subs={a: params[0], b: params[1]})
#     gradient_y = diff(f(a, b), b).evalf(subs={a: params[0], b: params[1]})
#     gradients = np.array([gradient_x, gradient_y])
#     params = params - multiplier * gradients
#
# print("the values in gradient array", gradients)
# print("min occurs at x value of:", params[0])
# print("min occurs at y value of:", params[1])
# print("the cost is:", f(params[0], params[1]))

#  graphing 3D gradient descent and advanced NumPy arrays

multiplier = 0.1
max_iter = 200
params = np.array([1.8, 1.0])  # initial guess
values_array = params.reshape(1, 2)


for n in range(max_iter):
    gradient_x = diff(f(a, b), a).evalf(subs={a: params[0], b: params[1]})
    gradient_y = diff(f(a, b), b).evalf(subs={a: params[0], b: params[1]})
    gradients = np.array([gradient_x, gradient_y])
    params = params - multiplier * gradients
    # values_array = np.append(values_array, params.reshape(1, 2), axis=0)
    values_array = np.concatenate((values_array, params.reshape(1, 2)), axis=0)

print("the values in gradient array", gradients)
print("min occurs at x value of:", params[0])
print("min occurs at y value of:", params[1])
print("the cost is:", f(params[0], params[1]))
# generating 3d plot

fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')
ax.set_xlabel("x", fontsize=20)
ax.set_ylabel("y", fontsize=20)
ax.set_zlabel("f(x, y) - cost", fontsize=20)
ax.plot_surface(x, y, f(x, y), cmap=cm.coolwarm, alpha=0.4)
ax.scatter(values_array[:,0], values_array[:,1], f(values_array[:,0], values_array[:,1]), s= 50, color='red')

plt.show()
# advanced umpy array practice

kirk = np.array([['captain', 'guitar']])
hs_band = np.array([['black though','mc'],['quest love','drums']])

the_root = np.append(arr=hs_band, values=kirk, axis=0)
print(the_root)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x = np.array([[0.1, 1.2, 2.4, 3.2, 4.1, 5.7, 6.5]]).transpose()
y = np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape(7, 1)

print("shape of x array:", x.shape)
print("shape of y array:", y.shape)

regr = LinearRegression()
regr.fit(x, y)

print('theta 0:', regr.intercept_[0])
print('theta 1:', regr.coef_[0][0])

plt.scatter(x, y, s=50)
plt.plot(x, regr.predict(x), color='orange', linewidth= 3)
plt.xlabel('x values')
plt.ylabel('y values')

plt.show()

y_hat = 0.847535148602954 + 1.2227264637835913 * x
print('est values y_hat are:\n', y_hat)
print('in comparison, the actual y values are: \n', y)

def MSE(y, y_hat):
    # mse_calc = 1/7 * sum((y - y_hat) ** 2)
    mse_calc = (1/y.size) * sum((y-y_hat)**2)
    # mse_calc = np.average(((y - y_hat) ** 2), axis=0)
    return mse_calc

print('Manually calculated MSE is:', MSE(y, y_hat))
print('MSE regression using manual calc is:', mean_squared_error(y, y_hat))
print('MSE regression is:', mean_squared_error(y, regr.predict(x)))

#  3D plot for the MSE cost function
nr_thetas = 200
th_0 = np.linspace(start=-1, stop=3, num=nr_thetas)
th_1 = np.linspace(start=-1, stop=3, num=nr_thetas)
plot_t0, plot_t1 = np.meshgrid(th_0, th_1)

plot_cost = np.zeros((nr_thetas, nr_thetas))

for i in range(nr_thetas):
    for j in range(nr_thetas):
        y_hat = plot_t0[i][j] + plot_t1[i][j] * x
        plot_cost[i][j] = MSE(y, y_hat)

print('shape of plot_t0:', plot_t0.shape)
print('shape of plot_t1:', plot_t1.shape)
print('shape of plot_cost:', plot_cost.shape)

# plotting MSE
fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('cost - MSE', fontsize=20)

ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.hot)

plt.show()

# print('Min vale of plot_cost:', plot_cost.min())
# ij_min = np.unravel_index(indices=plot_cost.argmin(), dims=plot_cost.shape)
# print('min occurs at (i, j):', ij_min)
# print('min mse for theta 0 at plot_t0[111][91]', plot_t0[111][91])
# print('min mse for theta 1 at plot_t0[111][91]', plot_t1[111][91])


# nested loops practice
for i in range(3):
    for j in range(3):
        print(f'value of i is {i} and j is {j}')

# MSE and Gradient descent
# inputs: x_values, y_values, array of theta parameters (theta0 at index 0 and theta1 at index 1)
def grad(x, y, thetas):
    n = y.size
    theta0_slope = (-2/n) * sum(y-thetas[0] - thetas[1]*x)
    theta1_slope = (-2 / n) * sum((y - thetas[0] - thetas[1] * x)*x)

    return np.array([theta0_slope[0], theta1_slope[0]])

multiplier = 0.01
thetas = np.array([2.9, 2.9])

# collect data points for scatter plot
plot_vals = thetas.reshape(1, 2)
mse_vals = MSE(y, thetas[0]+thetas[1]*x)


for i in range(1000):
    thetas = thetas - multiplier * grad(x, y, thetas)
#     append the new values to our numpy arrays
    plot_vals = np.concatenate((plot_vals, thetas.reshape(1, 2)), axis = 0)
    mse_vals = np.append(arr=mse_vals, values=MSE(y, thetas[0]+thetas[1]*x))

print('Min occurs at theta 0:', thetas[0])
print('Min occurs at theta 1:', thetas[1])
print('MSE is:', MSE(y, thetas[0]+thetas[1]*x))

# plotting MSE
fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')

ax.set_xlabel('Theta 0', fontsize=20)
ax.set_ylabel('Theta 1', fontsize=20)
ax.set_zlabel('cost - MSE', fontsize=20)

ax.scatter(plot_vals[:, 0], plot_vals[:, 1], mse_vals, s=80, color='black')
ax.plot_surface(plot_t0, plot_t1, plot_cost, cmap=cm.rainbow, alpha=0.4)

plt.show()















































