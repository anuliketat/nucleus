import numpy as np
import pandas as pd

def mape(y_true, y_pred): #Mean Absolute Percentage Error
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def mase(actual, forecast): #Mean Absolute Scaled Error
    myData = pd.DataFrame({'forecast':forecast, 'actual':actual})
    n = len(myData)
    naive_diff = []
    for i in range(len(actual)):
        if i == 0:
            temp = actual[i]
        else:
            temp = actual[i]-actual[i-1]
            temp = abs(temp)
        naive_diff.append(temp)
    errors = myData['actual'] - myData['forecast']
    scaledErrors = abs(errors)/(abs(sum(naive_diff))*(1/(n-1)))
    mase = (abs(scaledErrors).sum())/n
    return mase