
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from src.data_ingest import IngestData
from src.visual import show_hist
from src.math_op import show_skew_kurt


if __name__ == "__main__":
    data_load=IngestData("./data/housing.csv")
    data=data_load.open_csv()
    print(data.columns)

    #correlation 
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    
    #outlier checking and log transformation 
    show_hist(data['price'])
    show_skew_kurt(data['price'])
    qq_plot(data['price'])
    data["price"]=transform_root(data["price"])
    qq_plot(data['price'])
    
    #Split train/test
    X = data.drop("price", axis=1)
    y = data["price"]  # I
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

     #Fit regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    #Make predictions
    predictions = lr.predict(X_test)
    print("Actual value of the house: - ", y_test[0])
    print("Model Predicted Value: - ", predictions[0])
    
    #Evaluate model 
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    rsquare=r2_score(y_test, predictions)
    print(mse)
    
    #Analyze residuals
    y_train_pred = lr.predict(X_train)
    residuals = y_train - y_train_pred
    show_hist_residuals(residuals)

    #Homoscedasticity check 
    plt.scatter(
        x=y_train_pred,
        y=residuals
                )
    
    plt.axhline(y=0, linestyle="--")
    plt.title("Homoscedasticity Check")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()
    #Normality check
    qq_plot(residuals)
    #Independence check 
    plt.plot(residuals)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals Over Observations")
    plt.xlabel("Observation")
    plt.ylabel("Residual")
    plt.show()
    
    


