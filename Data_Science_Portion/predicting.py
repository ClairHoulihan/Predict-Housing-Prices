#!/usr/bin/env python3

##########################################################################################
########################### Initial imports and functions ################################
##########################################################################################

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from exceptions import *
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
        (optional) max_nums - The array that we are adding max values to

output: An array that contains an i number of maximum values from oldArray
'''
def pop_max_ammount(oldArray, i, max_nums=[]):
        
    if i <= 0:
        return max_nums
    else:
        max_nums.append(oldArray.pop(oldArray.index(max(oldArray))))
        return pop_max_ammount(oldArray, i-1, max_nums=max_nums)
    
'''
pop_min_ammount pops out the min number from an array and adds it to a new array (by default, can add to an existing 
array if the user so chooses). The amount of min values taken from the old array is determined by i

input: oldArray - The array the function is extracting min values from
       i - The number of min values we are extracting from the old array
       (optional) min_nums - The array that we are adding min values to

output: An array that contains an i number of minimum values from oldArray
'''
def pop_min_ammount(oldArray, i, min_nums=[]):

    if i <= 0:
        return min_nums
    else:
        min_nums.append(oldArray.pop(oldArray.index(min(oldArray))))
        return pop_min_ammount(oldArray, i-1, min_nums=min_nums)

'''
Housing_Model is a class that handles all of the processing on two sets of data, 
one from train.csv that will retrieve the sales price of each home, and also include 
the columns that influence it, while test.csv has every column from train.csv except
for the sales price column. The functions contained within it handle the preprocessing,
modeling, and the plotting of each model.

input: (optional) trainingSet - The data on which to train the model
(optional) testSet - The data on which to test the model
(optional) columns - The columns that are to be looked at for analysis

output: None
'''
class Housing_Model():
    def __init__(self, training_set=pd.read_csv("/home/clair/Housing_Project/Data_Science_Portion/house-prices-advanced-regression-techniques/train.csv"),
    test_set=pd.read_csv("/home/clair/Housing_Project/Data_Science_Portion/house-prices-advanced-regression-techniques/test.csv"), columns=None):
        self.training_set = training_set
        self.test_set = test_set

        if columns == None:
            self.columns = list(training_set.columns)
        else:
            self.columns = columns

        columns_to_drop = []

        for col in test_set.columns:
            if col not in columns:
                columns_to_drop.append(col)

        self.training_set = self.training_set.drop(columns=columns_to_drop)
        self.test_set = self.test_set.drop(columns=columns_to_drop)

    ##########################################################################################
    ########################### Preprocessing of the data ####################################
    ##########################################################################################
    '''
    preprocessing does all of the data clean up that is necessary, primarily converting all of 
    the non-numeric values into numeric values and filling in NaN values with 0

    input: None

    output: None
    '''
    def preprocessing(self):

        for col in self.columns:
            if col == "SalePrice":
                continue
            for i in range(0, len(self.training_set.index)):
                if not is_number(self.training_set[col][i]):
                    self.training_set = pd.concat([self.training_set, pd.get_dummies(self.training_set[col], dtype=float, prefix=col, dummy_na=True)], axis=1)
                    self.test_set = pd.concat([self.test_set, pd.get_dummies(self.training_set[col], dtype=float, prefix=col, dummy_na=True)], axis=1)
                    self.training_set.drop([col], axis=1, inplace=True)
                    self.test_set.drop([col], axis=1, inplace=True)
                    break
        
        self.training_set.fillna(0, inplace=True)
        self.test_set.fillna(0, inplace=True)

        self.predictors = []

        for col in self.training_set.columns:
            if col == "SalePrice":
                continue
            else:
                self.predictors.append(col)
        
    ##########################################################################################
    ########################### Modeling the data ############################################
    ##########################################################################################

    '''
    model sets the model for the future plots to be using a regression type specified by the user 
    and fits the training and testing data to that model.

    input: reg_type - the type of regression to perform with the data (default is Linear)

    output: None
    '''
    def model(self, reg_type="Linear"):

        if reg_type == "Linear":
            self.selector = LinearRegression()
            self.reg_type = reg_type
        elif reg_type == "Ridge":
            self.selector = Ridge()
            self.reg_type = reg_type
        else:
            raise UnknownModelError(reg_type)

        try:
            self.train_x = self.training_set[self.predictors]
            self.train_y = self.training_set[["SalePrice"]]

            self.selector.fit(self.training_set[self.predictors], self.training_set["SalePrice"])
            self.test_x = self.test_set[self.predictors]
            self.test_y = self.selector.predict(self.test_x)

            self.ttest_y = self.selector.predict(self.train_x)

        except AttributeError:
            raise NoDataError



    ##########################################################################################
    ########################### Graphing the max coefficients ################################
    ##########################################################################################

    '''
    coeffs shows a bar chart of the columns that have the greatest or least influence on the y values
    in the model

    input: max_or_min - determine what are the max coefficients vs the min coefficients

    output: None
    '''
    def coeffs(self, max_or_min):
    
        try:
            self.coefficients = self.selector.coef_

        except AttributeError:
            raise NoSelectorError

        if max_or_min == "max":
            vals = pop_max_ammount(self.coefficients.tolist(), 5)
        elif max_or_min == "min":
            vals = pop_min_ammount(self.coefficients.tolist(), 5)
        else:
            raise UnexpectedValueError(max_or_min, ["max", "min"])

        ticks = []
        for num in vals:
            ticks.append(self.predictors[self.coefficients.tolist().index(num)])
            
        i = range(0, len(ticks))

        fig = plt.figure(figsize=(15, 10), dpi=80)

        plt.title(max_or_min.capitalize() + " Coefficients")
        plt.bar(i, vals)

        plt.xticks(i, ticks)

        fig.savefig(max_or_min + '_coefficients.png')

        plt.show()
        plt.clf()

    ##########################################################################################
    ########################### Predicted vs actual house values plot ########################
    ##########################################################################################

    '''
    plot plots the values from the predicted house values, and plots the values from the 
    actual house values

    input: None

    output: None
    '''
    def plot(self):

        try:
            fig = plt.figure()
            plt.title("Predicted Vs Actual House Values Using %s Regression" % (self.reg_type.lower()))
            plt.plot(range(0, 50), self.ttest_y[:50], 'b-o', label="Models predictions")
            plt.plot(range(0, 50), self.train_y[:50], 'g-o', label="Actual Values")
            plt.legend()

            fig.savefig('predicted_vs_actual_house_values_using_%s_regression' % (self.reg_type.lower()))

            plt.show()
            plt.clf()

        except AttributeError:
            raise NoSelectorError


    ##########################################################################################
    ########################### Predicted vs actual house values sorted plot #################
    ##########################################################################################

    '''
    sorted_plot plots the values from the predicted house values, and plots the values from the 
    actual house values in order by the actual house prices

    input: None

    output: None
    '''
    def sorted_plot(self):

        try:
            dataLimit = 50

            self.pltTtest_y = []
            self.pltTrain_y = []
            plt_sort = []

            trainnum_y = self.train_y.to_numpy()

            for i in range(0, dataLimit):
                plt_sort.append([self.ttest_y[i], trainnum_y[i]])
            
            sorted_plt = sorted(plt_sort, key=operator.itemgetter(1))
            
            for pltt in sorted_plt:
                self.pltTtest_y.append(pltt[0])
                self.pltTrain_y.append(pltt[1])
                
            x_ticks = []
            for i in range(0, dataLimit):
                x_ticks.append(i)

            fig = plt.figure()

            plt.title("Predicted Vs Actual House Values Using %s Regression" % (self.reg_type.lower()))
            plt.plot(x_ticks, self.pltTtest_y, 'b-o', label="Models predictions")
            plt.plot(x_ticks, self.pltTrain_y, 'g-o', label="Actual Values")
            plt.legend()

            fig.savefig('predicted_vs_actual_house_values_sorted_using_%s_regression' % (self.reg_type.lower()))

            plt.show()
            plt.clf()

        except AttributeError:
            raise NoSelectorError

    ##########################################################################################
    ########################### Evaluation Metrics ###########################################
    ##########################################################################################

    '''
    model_eval prints out some evaluation metrics in order to evaluate how well the model 
    performed with the dataset

    input: None

    output: None
    '''
    def model_eval(self):
            
        try:
            print("%s Regression Evaluation: " % (self.reg_type))
            print("mean_absolute_error: %.2f" % (mean_absolute_error(self.train_y, self.ttest_y)))
            print("mean_squared_error: %.2f" % (mean_squared_error(self.train_y, self.ttest_y)))
            print("r2_score: %.2f\n" % (r2_score(self.train_y, self.ttest_y)))

        except AttributeError:
            raise NoSelectorError


##########################################################################################
########################### Testing ######################################################
##########################################################################################

testing = Housing_Model(columns=["Street", "Neighborhood", "OverallQual", "OverallCond", "RoofStyle"])
testing.preprocessing()
testing.model(reg_type="Linear")
testing.coeffs("max")
testing.coeffs("min")
testing.plot()
testing.sorted_plot()
testing.model_eval()
testing.model(reg_type="Ridge")
testing.plot()
testing.sorted_plot()
testing.model_eval()
