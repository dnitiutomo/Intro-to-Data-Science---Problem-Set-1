# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:16:04 2014

@author: Daniel
"""

import numpy as np
import pandas
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import datetime
from ggplot import *

def plot_residuals(predictions,data):
    plt.figure()
    (data - predictions).hist(range=[-10000,10000],bins=20)
    """
    s_y = pandas.Series(predictions-data)
    s_x = pandas.Series(range(0,len(s_y))) 
    df = pandas.DataFrame({'residuals': s_y, 
                           'num': s_x})
    
    plt = ggplot(df, aes('num', 'residuals')) + geom_point() + ggtitle("Avg Entries by Variable") + xlab("num") + ylab("residuals")
    """    
    print plt

def mann_whit_test(dataframe,var):
    none = dataframe[dataframe[var]==0]['ENTRIESn_hourly']
    yes = dataframe[dataframe[var]==1]['ENTRIESn_hourly']
    
    none_mean = np.mean(none)
    yes_mean = np.mean(yes)
    
    print 'none: ', none_mean    
    print 'yes: ', yes_mean
    
    test = scipy.stats.mannwhitneyu(yes, none)
    
    print 'p-value: ', test[1]


def convert_to_day(row):
    return datetime.datetime.strptime(str(row['DATEn']),'%Y-%m-%d').strftime('%a')

def get_avg_for_unit(row,avgs):
    return avgs[str(row['UNIT'])]

def normalize_features(array):
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

def predictions_OLS(dataframe):
    X = dataframe.copy()
    y = X.pop('ENTRIESn_hourly')
    avgs = y.groupby(X.UNIT).mean()
    
    dataframe['UNIT_avg'] = dataframe.apply(lambda row: get_avg_for_unit(row,avgs),axis=1)
    dataframe['UNIT_ord'] = pandas.Categorical(dataframe.UNIT).labels
    
    dataframe['Day'] = df.apply(convert_to_day,1)  

    Y = dataframe['ENTRIESn_hourly']
    X = dataframe[['UNIT_avg','rain']]
    
    #dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')    
    #X = X.join(dummy_units)
    
    dummy_hour = pandas.get_dummies(dataframe['Hour'])
    X = X.join(dummy_hour)    
    
    dummy_days = pandas.get_dummies(dataframe['Day'])
    X = X.join(dummy_days)      
    #X = X.join(dataframe[['meantempi','fog']])
    
    X = sm.add_constant(X)  
    
    model = sm.OLS(Y,X)
    results = model.fit()
    print results.summary()
    prediction = results.predict(X)
    
    return prediction

def compute_cost(features, values, theta):

    pref = (2 * len(values))
    predicted = np.dot(features,theta)
    diff = predicted - values
    errors = np.square(diff)
    cost = np.sum(errors) / pref

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        temp_theta = theta
        feat_trans = features.transpose()
        predicted = np.dot(features, theta)
        diff = predicted - values
        
        j_prime = np.dot(feat_trans,diff) / m
        
        theta = temp_theta - (alpha*j_prime)

        cost_history.append(compute_cost(features, values, theta))
    return theta, pandas.Series(cost_history)

def predictions(dataframe):
    """
    X = dataframe.copy()
    y = X.pop('ENTRIESn_hourly')
    avgs = y.groupby(X.UNIT).mean()
    dataframe['UNIT_avg'] = dataframe.apply(lambda row: get_avg_for_unit(row,avgs),axis=1)
    dataframe['UNIT_avg_squared'] = dataframe['UNIT_avg']**2
    """
    dataframe['Day'] = df.apply(convert_to_day,1)
    
    features = dataframe[[]]
    
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')    
    features = features.join(dummy_units)
    
    dummy_hour = pandas.get_dummies(dataframe['Hour'])
    features = features.join(dummy_hour)    
    
    dummy_days = pandas.get_dummies(dataframe['Day'])
    features = features.join(dummy_days)

    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept), this is for the constant
    
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    alpha = 1
    num_iterations = 200

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)
    
    plot = None

    #plot = plot_cost_history(alpha, cost_history)
    print theta_gradient_descent
    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions


def plot_cost_history(alpha, cost_history):
   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )

def compute_r_squared(data, predictions):

    mean = np.mean(data)
    
    SSres = np.sum((data-predictions)**2)
    SStot = np.sum((data-mean)**2)
    
    r_squared = 1 - (SSres/SStot)
    
    return r_squared

df = pandas.read_csv(r'C:\Users\Daniel\Documents\OnlineLearning\Intro to DS\Github\turnstile_data_master_with_weather.csv')
#mann_whit_test(df,'fog')


grad_descent = predictions(df)
ols = predictions_OLS(df)

plot_residuals(grad_descent,df['ENTRIESn_hourly'])

#print compute_r_squared(df['ENTRIESn_hourly'],grad_descent)
print compute_r_squared(df['ENTRIESn_hourly'],ols)