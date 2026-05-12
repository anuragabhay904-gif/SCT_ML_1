import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dataset was loaded 
data = pd.read_csv('train.csv')

# input features for the predicition
features = [
    'GrLivArea',
    'BedroomAbvGr',
    'FullBath',
    'HalfBath',
    'TotRmsAbvGrd',
    'GarageCars',
    'GarageArea',
    'OverallQual',
    'YearBuilt'
]

# feature for the output
target = 'SalePrice'

# Now a DataFrame is created with the input features and the targeted value 
df = data[features + [target]]

# removing all the missing value using the dropna method
df = df.dropna()

# arranging the input and output features 
X = df[features]
y = df[target]

# we will split the data into training and the testing data to train the model
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# now we have split the data , we have the inputs(features) and we know the value we have to predict(SalePrice)
# All the missing values has been removed already 
# lets train the model now to have hypotheses to get the output
from sklearn.linear_model import LinearRegression

model = LinearRegression()
# we have created a model ,lets train the model.
model.fit(X_train,y_test)

# Training is done , lets predict the value now 
predict = model.predict(X_test)

new_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2],
    'HalfBath': [1],
    'TotRmsAbvGrd': [7],
    'GarageCars': [2],
    'GarageArea': [500],
    'OverallQual': [7],
    'YearBuilt': [2005]
})

predicted_price = model.predict(new_house)

print("\n===== NEW HOUSE PREDICTION =====")
print("Predicted House Price:", predicted_price[0])


