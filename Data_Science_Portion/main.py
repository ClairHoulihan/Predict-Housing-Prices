#!/usr/bin/env python3

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''
is_number checks to see if the input can be converted into a float, if it can, it will return True,
otherwise, the functions will return false.

input:  s - Any object

output: True or False

'''
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

'''
pop_max_ammount pops out the max number from an array and adds it to a new array (by default, can add to an existing 
array if the user so chooses). The amount of max values taken from the old array is determined by i

input:  oldArray - The array the function is extracting max values from
        i - The number of max values we are extracting from the old array
        max_nums - The array that we are adding max values to

output: An array that contains an i number of maximum values from oldArray
'''
def pop_max_ammount(oldArray, i, max_nums=[]):
        
    if i <= 0:
        return max_nums
    else:
        max_nums.append(oldArray.pop(oldArray.index(max(oldArray))))
        return pop_max_ammount(oldArray, i-1, max_nums=max_nums)
    
'''
pop_min_ammount pops out the min number from an array and adds it to a new array
'''
def pop_min_ammount(oldArray, i, min_nums=[]):

    if i <= 0:
        return min_nums
    else:
        min_nums.append(oldArray.pop(oldArray.index(min(oldArray))))
        return pop_min_ammount(oldArray, i-1, min_nums=min_nums)
    

##########################################################################################

trainingSet = pd.read_csv("./house-prices-advanced-regression-techniques/train.csv")
testSet = pd.read_csv("./house-prices-advanced-regression-techniques/test.csv")

columns = list(trainingSet.columns)

trainingSetEncoded = trainingSet
testSetEncoded = testSet

for col in columns:
    if col == "SalePrice":
        continue
    for i in range(0, len(trainingSet.index)):
        if not is_number(trainingSet[col][i]):
            trainingSetEncoded = pd.concat([trainingSetEncoded, pd.get_dummies(trainingSet[col], dtype=float, prefix=col, dummy_na=True)], axis=1)
            testSetEncoded = pd.concat([testSetEncoded, pd.get_dummies(trainingSet[col], dtype=float, prefix=col, dummy_na=True)], axis=1)
            trainingSetEncoded.drop([col], axis=1, inplace=True)
            testSetEncoded.drop([col], axis=1, inplace=True)
            break
    

trainingSetEncoded.fillna(0, inplace=True)
testSetEncoded.fillna(0, inplace=True)

predictors = []

for col in trainingSetEncoded.columns:
    if col == "SalePrice":
        continue
    else:
        predictors.append(col)

msk1 = np.random.rand(len(trainingSetEncoded)) < 0.8
msk2 = np.random.rand(len(testSetEncoded)) < 0.8
train = trainingSetEncoded[msk1]
test = testSetEncoded[msk2]
        
##########################################################################################
    
#print(trainingSetEncoded.columns)

selector = LinearRegression()
train_x = train[predictors]
train_y = train[["SalePrice"]]

selector.fit(trainingSetEncoded[predictors], trainingSetEncoded["SalePrice"])
test_x = test[predictors]
test_y = selector.predict(test_x)

ttest_y = selector.predict(train_x)

##########################################################################################

coefficients = selector.coef_

max5 = pop_max_ammount(coefficients.tolist(), 5)
ticks = []

coeffList = coefficients.tolist()

for num in max5:
    ticks.append(predictors[coefficients.tolist().index(num)])
    
i = range(0, len(ticks))

fig = plt.figure(figsize=(15, 10), dpi=80)

plt.title("Max Coefficients")
plt.bar(i, max5)

plt.xticks(i, ticks)

fig.savefig('max_coefficients.png')

plt.show()
plt.clf()

##########################################################################################


min5 = pop_min_ammount(coefficients.tolist(), 5)
ticks = []

coeffList = coefficients.tolist()

for num in min5:
    ticks.append(predictors[coefficients.tolist().index(num)])
    
i = range(0, len(ticks))

fig = plt.figure(figsize=(15, 10), dpi=80)

plt.title("Min Coefficients")
plt.bar(i, min5)

plt.xticks(i, ticks)

fig.savefig('min_coefficients.png')

plt.show()
plt.clf()

##########################################################################################

fig = plt.figure()
plt.title("Predicted Vs Actual House Values")
plt.plot(range(0, 50), ttest_y[:50], 'b-o', label="Models predictions")
plt.plot(range(0, 50), train_y[:50], 'g-o', label="Actual Values")
plt.legend()

fig.savefig('predicted_vs_actual_house_values')

plt.show()
plt.clf()

##########################################################################################

dataLimit = 50

pltTtest_y = []
pltTrain_y = []
pltSort = []

trainnum_y = train_y.to_numpy()

for i in range(0, dataLimit):
    pltSort.append([ttest_y[i], trainnum_y[i]])
        
sorted_plt = sorted(pltSort, key=operator.itemgetter(1))
    
for pltt in sorted_plt:
    pltTtest_y.append(pltt[0])
    pltTrain_y.append(pltt[1])
    
x_ticks = []
for i in range(0, dataLimit):
    x_ticks.append(i)

fig = plt.figure()

plt.title("Predicted Vs Actual House Values")
plt.plot(x_ticks, pltTtest_y, 'b-o', label="Models predictions")
plt.plot(x_ticks, pltTrain_y, 'g-o', label="Actual Values")
plt.legend()

fig.savefig('predicted_vs_actual_house_values_sorted')

plt.show()
plt.clf()

##########################################################################################

print("mean_absolute_error: %.2f" % (mean_absolute_error(pltTrain_y, pltTtest_y)))

print("mean_squared_error: %.2f" % (mean_squared_error(pltTrain_y, pltTtest_y)))

print("r2_score: %.2f" % (r2_score(pltTrain_y, pltTtest_y)))

##########################################################################################

selector = Ridge()
train_x = train[predictors]
train_y = train[["SalePrice"]]

selector.fit(trainingSetEncoded[predictors], trainingSetEncoded["SalePrice"])
test_x = test[predictors]
test_y = selector.predict(test_x)

ttest_y = selector.predict(train_x)

##########################################################################################

fig = plt.figure()
plt.title("Predicted Vs Actual House Values")
plt.plot(range(0, 50), ttest_y[:50], 'b-o', label="Models predictions")
plt.plot(range(0, 50), train_y[:50], 'g-o', label="Actual Values")
plt.legend()

fig.savefig('ridge_predicted_vs_actual_house_values')

plt.show()
plt.clf()

##########################################################################################

dataLimit = 50

pltTtest_y = []
pltTrain_y = []
pltSort = []

trainnum_y = train_y.to_numpy()

for i in range(0, dataLimit):
    pltSort.append([ttest_y[i], trainnum_y[i]])
        
sorted_plt = sorted(pltSort, key=operator.itemgetter(1))
    
for pltt in sorted_plt:
    pltTtest_y.append(pltt[0])
    pltTrain_y.append(pltt[1])
    
x_ticks = []
for i in range(0, dataLimit):
    x_ticks.append(i)

fig = plt.figure()

plt.title("Predicted Vs Actual House Values")
plt.plot(x_ticks, pltTtest_y, 'b-o', label="Models predictions")
plt.plot(x_ticks, pltTrain_y, 'g-o', label="Actual Values")

fig.savefig('ridge_predicted_vs_actual_house_values_sorted')

plt.legend()
plt.show()

##########################################################################################

print("mean_absolute_error: %.2f" % (mean_absolute_error(pltTrain_y, pltTtest_y)))

print("mean_squared_error: %.2f" % (mean_squared_error(pltTrain_y, pltTtest_y)))

print("r2_score: %.2f" % (r2_score(pltTrain_y, pltTtest_y)))
