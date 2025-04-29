import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("email_phishing_data_othermodel.csv")
x = data[["num_words","num_unique_words","num_stopwords","num_links","num_unique_domains","num_email_addresses","num_spelling_errors","num_urgent_keywords"]].values
y = data["label"].values

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= .2)

model = LinearRegression().fit(xtrain,ytrain)

coef = np.around(model.coef_, 2)
intecerpt = round(float(model.intercept_), 2)
r_squared = round(model.score(x,y), 2)

print("The Coefficient:", coef)
print(f"Model's Linear Equation: y= {coef[0]}x + {coef[1]}x1 + {coef[2]}x2 + {coef[3]}x3 + {coef[4]}x4 + {coef[5]}x5 + {coef[6]}x6 + {coef[7]}x7 + {intecerpt}")
print("R Squared Value:", r_squared)

print("***************")
print("Testing Results")

predict = model.predict(xtest)
predict = np.around(predict, 2)
print(predict)