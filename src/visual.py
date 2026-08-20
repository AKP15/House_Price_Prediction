import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm,skew
from sklearn.model_selection import train_test_split             from sklearn.linear_model import LinearRegression


def show_hist(data:pd.DataFrame):
    plt.hist(data)
    plt.title("price frequency distribution")
    plt.xlabel("price")
    plt.ylabel("frequency")
    plt.show()


def show_hist_residuals(data:pd.DataFrame):
    plt.hist(data)
    plt.title("residuals check")
    plt.xlabel("residuals")
    plt.ylabel("frequency")
    plt.show()

def qq_plot(data:pd.DataFrame):
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot")
    plt.show()


    
