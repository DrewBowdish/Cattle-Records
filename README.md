# Cattle-Records
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm,probplot
df = pd.read_csv("Cattle 2019 github.csv")
df.head()
df.describe()
df.tail()
plt.scatter(df["5/15-Wt/DofA"],df["5/15-205Wt."])
X = np.array(df["5/15-Wt/DofA"]).reshape(-1,1)
Y = np.array(df["5/15-205Wt."]).reshape(-1,1)
OLS = LinearRegression()
OLS.fit(X,Y)
yhat = OLS.predict(X)
plt.scatter(X,Y)
plt.plot(X,yhat,color='red')
OLS.score(X,Y)
OLS.coef_
30*OLS.coef_ + OLS.intercept_
OLS.intercept_
plt.hist(Y-yhat)
