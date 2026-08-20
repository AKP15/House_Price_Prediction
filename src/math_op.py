import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm,skew
from sklearn.model_selection import train_test_split             from sklearn.linear_model import LinearRegression


def show_skew_kurt(data:pd.DataFrame):
    print("skew %f" % data['price'].skew())
    print("kurt %f" % data['price'].kurt())

def transform_root(data:pd.DataFrame):
    return np.log1p(data)
